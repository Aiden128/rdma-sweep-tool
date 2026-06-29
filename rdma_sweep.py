#!/usr/bin/env python3
"""RDMA parameter sweep tool for perftest.

Operations:
  1. rdma_sweep.py -c config.yaml -o results/     # run sweep
  2. rdma_sweep.py --report results/               # generate SVG report
"""

from __future__ import annotations

import argparse
import csv
import html
import itertools
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, NamedTuple, NoReturn, cast

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]

from rdma_config import (
    _ensure_int,
    _str_field,
    parse_bool,
    resolve_perftest_paths,
    runtime_config as _runtime_config,
)
from rdma_remote import RemoteResult, init_local_hosts, run_remote_result as _run_remote_result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def timestamp() -> str:
    """ISO-8601 UTC timestamp, second precision (Python 3.12 compat)."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_size(s: str) -> int:
    """Parse human-readable sizes like '1M', '64K', '2G' → bytes."""
    s = s.strip().upper()
    multipliers = {"K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4, "B": 1}
    for suffix, mul in multipliers.items():
        if s.endswith(suffix):
            return int(float(s[:-1]) * mul)
    return int(s)


def _config_val(config: dict[str, Any], key: str, default: Any = None) -> Any:
    """Get a config value, returning *default* when *key* is absent **or** ``None``.

    ``dict.get(key, default)`` returns *default* only when *key* is *absent*;
    a YAML null stored under *key* yields ``None`` instead.  This helper
    collapses both cases so callers don't need ``config.get(key, {}) or {}``.

    Note: *default* is returned as-is (no copy), so a caller that passes a
    shared mutable default would alias it.  Every call site passes a fresh
    literal (``[]`` / ``{}`` / a scalar), so this is safe in practice.
    """
    v = config.get(key)
    return default if v is None else v


def _record_command(
    process: dict[str, Any],
    label: str,
    result: RemoteResult,
) -> RemoteResult:
    """Record a remote-command result in ``process["commands"]`` and return it.

    Every call to ``_run_remote_result`` should be paired with
    ``process["commands"]["label"] = result.to_dict()`` so the process tracking
    dict has a forensic record of every SSH command.  This helper eliminates
    the repetition and makes it impossible to forget ``.to_dict()``.

    Because the function *returns* *result*, callers should nest the
    ``_run_remote_result`` call directly as the *result* argument and
    capture the return value in a single expression::

        log = _record_command(process, "server_log_tail", _run_remote_result(...))
    """
    process["commands"][label] = result.to_dict()
    return result


def _read_json(path: Path) -> Any:
    """Read and parse a JSON file.

    Args:
        path: Path to the JSON file.

    Returns:
        Parsed JSON data as a Python object.

    Raises:
        FileNotFoundError: *path* does not exist.
        json.JSONDecodeError: *path* contains invalid JSON.
    """
    return json.loads(path.read_text())


def _write_json(path: Path, data: Any, indent: int = 2) -> None:
    """Serialize *data* as JSON and write to *path*.

    Centralises ``path.write_text(json.dumps(...))`` so that the output
    format (indentation, encoding) is consistent across all call sites.

    Args:
        path: Destination file path.
        data: JSON-serialisable Python object.
        indent: Indentation level for the JSON output (default 2).
    """
    path.write_text(json.dumps(data, indent=indent))


def _get_dict(d: dict[str, Any], key: str, *fallback_keys: str) -> dict[str, Any]:
    """Extract a dict value, returning ``{}`` on missing/``None`` keys.

    Handles the common pattern ``d.get(key, {}) or {}`` which guards
    against both absent keys (where ``.get()`` returns the default) and
    YAML-backed ``None`` values (where ``None or {}`` yields ``{}``).

    Accepts optional *fallback_keys* for chained lookups such as
    ``d.get(a, d.get(b, {})) or {}``.

    .. note::

       Returns a **live reference** to the dict value in *d*, not a copy.
       Callers that mutate the result will affect the source dict.
       This mirrors ``dict.get()`` semantics and avoids allocation
       overhead on the hot path.
    """
    for k in (key,) + fallback_keys:
        v = d.get(k)
        if isinstance(v, dict):
            return v
    return {}

def _config_int(config: dict[str, Any], key: str, default: int = 0) -> int:
    """Extract an integer config value, returning *default* when absent/None.

    Delegates to ``_ensure_int`` from ``rdma_config`` for consistent
    YAML-``None`` handling across all config-reading helpers.
    """
    return _ensure_int(config, key, default)


def format_size(n: int) -> str:
    """Bytes → human-readable string."""
    if n < 0:
        return f"-{format_size(abs(n))}"
    value = float(n)
    for unit in ("B", "K", "M", "G"):
        if value < 1024:
            return f"{value:.1f}{unit}" if unit != "B" else f"{int(value)}{unit}"
        value /= 1024
    return f"{value:.1f}T"


def _parse_int_fields(parts: list[str]) -> list[int] | None:
    """Parse ``parts[1:]`` as integers, returning ``None`` on failure.

    Shared by ``_parse_proc_stat`` and ``_parse_proc_softirqs``, both of
    which iterate over /proc lines and skip malformed entries rather
    than aborting the sweep.
    """
    try:
        return [int(v) for v in parts[1:]]
    except ValueError:
        return None


def _parse_proc_stat(lines: list[str]) -> dict[str, dict[str, int]]:
    """Parse ``/proc/stat`` lines into per-core CPU tick dicts."""
    cores: dict[str, dict[str, int]] = {}
    for line in lines:
        parts = line.split()
        if not parts:
            continue
        if parts[0].startswith("cpu"):
            vals = _parse_int_fields(parts)
            if vals is None:
                continue
            if len(vals) < 8:  # zip would silently truncate, missing "idle"
                continue
            cores[parts[0]] = {k: v for k, v in
                               zip(["user", "nice", "system", "idle", "iowait",
                                    "irq", "softirq", "steal"], vals)}
    return cores


def _parse_proc_softirqs(lines: list[str]) -> dict[str, int]:
    """Parse ``/proc/softirqs`` lines into per-type summed interrupt counts."""
    softirqs: dict[str, int] = {}
    for line in lines:
        parts = line.split()
        if not parts:
            continue
        if parts[0].startswith("CPU"):
            # header line "CPU0 CPU1 CPU2 ..." — skip
            continue
        # line like "NET_RX:  12345  0  67890  0  ..."
        name = parts[0].rstrip(":")
        vals = _parse_int_fields(parts)
        if vals is None:
            continue
        softirqs[name] = sum(vals)
    return softirqs


def _parse_proc_meminfo(lines: list[str]) -> dict[str, int]:
    """Parse ``/proc/meminfo`` lines into key → kB dict."""
    mem: dict[str, int] = {}
    for line in lines:
        key, sep, rest = line.partition(":")  # single-pass, colon-safe
        if not sep:
            continue
        fields = rest.split()
        if not fields:
            continue  # "Key:" with no value — skip, don't IndexError-abort
        try:
            mem[key.strip()] = int(fields[0])
        except ValueError:
            continue  # skip a malformed line, don't abort the sweep
    return mem


def _mem_used_kb(m: dict[str, int]) -> int:
    """Compute used memory in kB from a parsed ``/proc/meminfo`` dict.

    Used = ``MemTotal - MemFree - Buffers - Cached`` — the conventional
    Linux "actually consumed" figure that excludes reclaimable page/buffer
    cache.  Missing keys default to 0 so a truncated/older meminfo degrades
    gracefully instead of raising ``KeyError``.

    Shared by ``SysMonitor.extract_mem`` (absolute used) and
    ``SysMonitor.compute_mem_delta`` (used-after minus used-before).
    """
    return (
        m.get("MemTotal", 0) - m.get("MemFree", 0)
        - m.get("Buffers", 0) - m.get("Cached", 0)
    )


# ---------------------------------------------------------------------------
# CPU / Memory monitor (runs via /proc on the remote host)
# ---------------------------------------------------------------------------

class SysMonitor:
    """Reads /proc/stat (per-core), /proc/softirqs, and /proc/meminfo.

    We capture a *pair* of samples (before / after) so we can compute
    per-core utilisation without the overhead of a persistent thread.
    The SSH-based test runner calls grab() before starting the test and
    on test completion; the difference is what matters.
    """

    PROC_STAT = "/proc/stat"
    PROC_MEMINFO = "/proc/meminfo"
    PROC_SOFTIRQS = "/proc/softirqs"

    def __init__(self, host: str, ssh_config: dict[str, Any] | None = None) -> None:
        self.host = host
        self.ssh_config = ssh_config

    def _run(self, cmd: str) -> list[str]:
        result = _run_remote_result(
            cmd, self.host, timeout=10,
            ssh_config=self.ssh_config, sudo=False,
        )
        if not result.ok:
            # run_remote_result never raises; surface the failure so grab()'s
            # try/except records it instead of silently returning empty samples
            # (which would make the *_sys_*_ok flags falsely report success).
            raise RuntimeError(result.error_summary())
        return result.stdout.splitlines()

    def grab(self) -> dict[str, Any]:
        """Return a snapshot of /proc/stat, /proc/softirqs, /proc/meminfo."""
        try:
            stat_lines = self._run(f"cat {self.PROC_STAT}")
            mem_lines = self._run(f"cat {self.PROC_MEMINFO}")
            sirq_lines = self._run(f"cat {self.PROC_SOFTIRQS}")
        except Exception as exc:
            return {"error": str(exc)}

        cores = _parse_proc_stat(stat_lines)
        softirqs = _parse_proc_softirqs(sirq_lines)
        mem = _parse_proc_meminfo(mem_lines)

        # The per-line guards above tolerate stray/malformed lines, but if the
        # read SUCCEEDED (rc 0) yet produced no parseable cpu or memory data at
        # all — truncated read, banner contamination, an unexpected /proc format
        # — the snapshot is useless.  Every Linux host's /proc/stat has cpu lines
        # and /proc/meminfo has MemTotal, so an empty result here is a failure,
        # not an idle host.  Return an error so it flows through the *_sys_*_ok
        # flag and the CSV "ERR" rendering instead of masquerading as a clean,
        # genuinely-idle data point (the symptom->cause trap this tool avoids).
        if not cores or not mem:
            return {
                "error": (
                    f"unparseable /proc on {self.host}: "
                    f"cores={len(cores)} mem_keys={len(mem)} "
                    "(read returned rc=0 but no usable cpu/mem data)"
                )
            }

        return {
            "time": timestamp(),
            "cores": cores,
            "mem_kB": mem,
            "softirqs": softirqs,
        }

    @staticmethod
    def compute_softirq_diff(
        before: dict[str, Any],
        after: dict[str, Any],
    ) -> dict[str, int]:
        """Return delta softirq counts (after - before) per softirq type."""
        be = _get_dict(before, "softirqs")
        ae = _get_dict(after, "softirqs")
        # only keys present in BOTH snapshots, so be[k] is always defined
        return {k: ae[k] - be[k] for k in ae if k in be}

    @staticmethod
    def compute_cpu_diff(
        before: dict[str, Any],
        after: dict[str, Any],
    ) -> dict[str, float]:
        """Return per-core utilisation (%) between two grabs."""
        result: dict[str, float] = {}
        be = _get_dict(before, "cores")
        ae = _get_dict(after, "cores")
        for key in be:
            if key not in ae:
                continue
            b = be[key]
            a = ae[key]
            b_total = sum(b.values())
            a_total = sum(a.values())
            dt = a_total - b_total
            if dt <= 0:
                result[key] = 0.0
            else:
                # Use .get(..., 0) so a /proc/stat cpu line with fewer than 4 fields
                # (e.g. a future kernel variant that drops the idle column) degrades
                # gracefully instead of crashing on KeyError.
                raw = 100.0 * (1 - (a.get("idle", 0) - b.get("idle", 0)) / dt)
                result[key] = max(0.0, min(100.0, raw))
        return result

    @staticmethod
    def extract_mem(after: dict[str, Any]) -> dict[str, str]:
        """Return key memory metrics from a grab (in human-readable form)."""
        m = _get_dict(after, "mem_kB")
        if not m:
            return {}
        return {
            "MemTotal":  format_size(m.get("MemTotal", 0) * 1024),
            "MemFree":   format_size(m.get("MemFree", 0) * 1024),
            "MemUsed":   format_size(_mem_used_kb(m) * 1024),
            "Buffers":   format_size(m.get("Buffers", 0) * 1024),
            "Cached":    format_size(m.get("Cached", 0) * 1024),
        }

    @staticmethod
    def compute_mem_delta(
        before: dict[str, Any],
        after: dict[str, Any],
    ) -> dict[str, str]:
        """Return delta memory (after - before) in human-readable form."""
        b = _get_dict(before, "mem_kB")
        a = _get_dict(after, "mem_kB")
        if not a or not b:
            return {}
        used_delta_kb = _mem_used_kb(a) - _mem_used_kb(b)
        return {
            "MemUsedDelta": format_size(used_delta_kb * 1024),
            "MemFreeDelta": format_size((a.get("MemFree", 0) - b.get("MemFree", 0)) * 1024),
        }


# ---------------------------------------------------------------------------
# Perftest runner
# ---------------------------------------------------------------------------

def _probe_port(
    host: str,
    port: int,
    ssh_config: dict[str, Any] | None,
    ss_sentinel: str,
) -> Any:
    """Run a single SSH port probe via ``ss`` on the remote host.

    Returns the SSH result object.  The remote command:
    - signals ``ss`` absence by printing *ss_sentinel* and exiting 0,
    - prints ``ready`` when the port is listening, and
    - exits 0 on any outcome so a non-ok result means transport failure.
    """
    return _run_remote_result(
        # Check ``ss`` exists first, then probe.  Without the existence
        # guard, a missing-ss failure is silently lumped with "port not
        # listening" -- both produce empty stdout at exit 0.
        f"command -v ss >/dev/null 2>&1 || {{ echo {ss_sentinel}; exit 0; }}; "
        f"ss -H -tln 'sport = :{int(port)}' 2>/dev/null | grep -q . && echo ready || true",
        host, ssh_config=ssh_config, sudo=False,
    )


def _wait_for_port(
    host: str,
    port: int,
    timeout: int = 30,
    ssh_config: dict[str, Any] | None = None,
) -> None:
    """Poll *host* until *port* is listening (or *timeout* seconds elapse).

    Distinguishes three outcomes on a reachable host:
      * **port up**       → returns immediately
      * **ss missing**    → times out with a clear diagnostic
      * **port not up**   → times out with ``did not listen``

    The probe exits 0 whenever SSH ran (via ``exit 0`` inside the guard
    for the ss-missing case, plus ``|| true`` beyond the pipeline for
    the port-not-up case), so a non-ok result unambiguously means the
    transport failed — NOT "no listener" — avoiding the symptom→cause
    mis-attribution this tool exists to prevent.
    """
    last_error = "no successful probe"
    reachable = False
    ss_missing = False
    _SS_SENTINEL = "SS_NOT_FOUND"
    for _ in range(timeout * 2):
        probe = _probe_port(host, port, ssh_config, _SS_SENTINEL)
        if probe.ok:
            reachable = True
            if "ready" in probe.stdout:
                return
            if probe.stdout.strip() == _SS_SENTINEL:
                ss_missing = True
                break
        else:
            last_error = probe.error_summary()
        time.sleep(0.5)
    if not reachable:
        raise TimeoutError(
            f"cannot reach {host} to probe port {port} within {timeout}s "
            f"(SSH/transport failure: {last_error})"
        )
    if ss_missing:
        raise TimeoutError(
            f"ss (socket statistics) not found on {host} — port {port} probe requires ss"
        )
    raise TimeoutError(f"{host}:{port} did not listen within {timeout}s")


def _env_prefix(perftest_config: dict[str, Any]) -> str:
    env = {
        str(k): str(v)
        for k, v in _get_dict(perftest_config, "env").items()
        if v is not None
    }
    rdma_core_lib = _str_field(perftest_config, "rdma_core_lib")
    if rdma_core_lib and "LD_LIBRARY_PATH" not in env:
        env["LD_LIBRARY_PATH"] = rdma_core_lib

    parts: list[str] = []
    for key, value in env.items():
        if not key.isidentifier():
            raise ValueError(f"invalid environment variable name: {key}")
        parts.append(f"{key}={shlex.quote(value)}")
    return " ".join(parts)


def _filtered_perftest_args(extra_args: list[str]) -> list[str]:
    filtered: list[str] = []
    skip_next = False
    for arg in extra_args:
        if skip_next:
            skip_next = False
            continue
        if arg in ("--out_json", "--out-json"):
            continue
        if arg in ("--out_json_file", "--out-json-file", "-D"):
            skip_next = True
            continue
        if (
            arg.startswith("--out_json_file")
            or arg.startswith("--out-json-file")
            or arg.startswith("-D")
        ):
            continue
        filtered.append(arg)
    return filtered


def _build_perftest_cmdline(
    extra_args: list[str], duration: int,
) -> tuple[str, str]:
    """Build the perftest argument string and duration flag.

    Strips any ``--out_json`` (``run_perftest`` adds its own), then returns
    ``(args_str, duration_arg)``.  *duration_arg* is an empty string when
    ``-n`` / ``--iters`` is already present (the server picks iteration
    count instead of wall-clock duration).
    """
    filtered = _filtered_perftest_args(extra_args)
    args_str = " ".join(shlex.quote(a) for a in filtered)
    duration_arg = "" if _has_flag(filtered, "-n", "--iters") else f"-D {int(duration)}"
    return args_str, duration_arg


def _has_flag(args: list[str], *flags: str) -> bool:
    """Check if any *flag* (e.g. ``-n``, ``--iters``) appears in *args*.

    Recognises bare flags, ``--flag=value``, and short-form attached values
    like ``-n1000`` so a user-provided iteration spec is never missed.
    """
    flag_set = set(flags)
    for arg in args:
        if arg in flag_set:
            return True
        for flag in flag_set:
            if arg.startswith(f"{flag}="):
                return True
            # Short flag with attached numeric value: ``-n1000`` (not ``-n 1000``).
            # Only match when the suffix is a digit string so ``-no_cma`` does
            # NOT trigger a false positive for ``-n``.
            if not flag.startswith("--") and arg.startswith(flag) and len(arg) > len(flag):
                rest = arg[len(flag):]
                if rest.isdecimal():
                    return True
    return False


def _port_from_args(extra_args: list[str], default_port: int) -> int:
    """Return the perftest port from *extra_args*, else *default_port*.

    Recognises every form perftest/getopt accepts — ``-p N``, ``-pN``,
    ``--port N``, ``--port=N`` — so the polled port always matches the port the
    server actually binds.  A mismatch would make _wait_for_port poll the wrong
    port and mis-report a healthy server as "did not listen" (a symptom->cause
    trap this tool exists to avoid).  Today _build_args only ever emits the
    ``-p <val>`` two-token form, but the parser is kept total so a future
    raw-arg passthrough cannot silently break port attribution.  An unparseable
    value is treated as absent (falls back to *default_port*) rather than
    raising and aborting the sweep mid-run.
    """
    n = len(extra_args)
    i = 0
    while i < n:
        arg = extra_args[i]
        val: str | None = None
        if arg in ("-p", "--port"):
            if i + 1 < n:
                val = extra_args[i + 1]
                i += 2
            else:
                i += 1
        elif arg.startswith("--port="):
            val = arg.split("=", 1)[1]
            i += 1
        elif arg.startswith("-p") and not arg.startswith("--") and len(arg) > 2:
            val = arg[2:]  # attached short form, e.g. "-p18515"
            i += 1
        else:
            i += 1
        if val is not None:
            try:
                return int(val)
            except ValueError:
                continue  # malformed value — keep scanning, else default
    return default_port


def _parse_perf_line(line: str) -> tuple[str, float] | None:
    """Parse one ``perf report`` output line into ``(key, self_pct)``.

    Returns ``None`` for comment lines, lines without ``[k]`` / ``[.]``
    annotation markers, or lines that fail to parse.  The caller
    (``_parse_perf_report``) aggregates the results into a dict.

    The self column is the second percentage field (first is children).
    Symbols from different DSOs sharing the same name are disambiguated
    as ``symbol@dso``.  Kernel DSO's key is the bare symbol name.
    """
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    # Top-level line: "  children%  self%  cmd  shared_obj  [annotation] symbol"
    # Must contain [k] or [.] annotation
    if "[k]" not in line and "[.]" not in line:
        return None
    parts = line.split()
    # Expected: children%, self%, cmd, shared_obj, [annotation], symbol...
    if len(parts) < 6:
        return None
    self_pct_str = parts[1].rstrip("%")
    try:
        self_pct = float(self_pct_str)
    except ValueError:
        return None
    if self_pct == 0.0:
        return None
    # Symbol may contain spaces (demangled C++ names).  Extract text
    # after the [k] or [.] annotation marker.
    #   parts[0..4] = children%, self%, cmd, shared_obj, [annotation]
    #   parts[5..]  = symbol tokens
    # The line-level guard above matches "[k]"/"[.]" as a SUBSTRING, but a
    # demangled symbol can embed those chars (e.g. "foo[.]bar") with no
    # standalone marker token.  Skip such a line rather than let a bare
    # next() raise StopIteration -- that would be caught upstream and
    # discard an otherwise-valid bandwidth+perf result for the whole run.
    marker_idx = next((i for i, p in enumerate(parts) if p in ("[k]", "[.]")), None)
    if marker_idx is None:
        return None
    dso = parts[3]
    symbol = " ".join(parts[marker_idx + 1:])
    # Some symbols (e.g., poll) can appear from different DSOs.
    # Always disambiguate with @dso so the key is unique.
    key = f"{symbol}@{dso}"
    # Shorten kernel DSO to keep keys readable.
    if dso == "[kernel.kallsyms]":
        key = symbol
    return key, self_pct


def _parse_perf_report(raw: str) -> dict[str, float]:
    """Parse ``perf report --stdio --no-header -g none`` into {symbol: self_pct}.

    Delegates per-line parsing to ``_parse_perf_line``.
    """
    result: dict[str, float] = {}
    for line in raw.splitlines():
        parsed = _parse_perf_line(line)
        if parsed is not None:
            key, self_pct = parsed
            result[key] = self_pct
    return result


def _fetch_server_log_tail(
    server_host: str,
    safe_log_path: str,
    ssh_config: dict[str, Any] | None,
    process: dict[str, Any],
) -> str:
    """Fetch the last 200 lines of the perftest server log for diagnostics."""
    log = _record_command(process, "server_log_tail", _run_remote_result(
        f"tail -200 {safe_log_path} 2>/dev/null || true",
        server_host,
        ssh_config=ssh_config,
        sudo=False,
    ))
    return log.stdout.strip()


def _enrich_error_with_server_log(
    result: dict[str, Any],
    server_host: str,
    server_log_q: str,
    ssh_config: dict[str, Any] | None,
) -> None:
    """If *result* carries an error, fetch the perftest server log tail.

    The server log (stdout/stderr of the perftest server process) often
    contains the root-cause of a client-side failure — negotiation mismatches,
    device errors, driver backtraces — that is invisible in the client's JSON
    output.  This is a no-op when *result* has no ``"error"`` key.
    """
    if "error" not in result:
        return
    process = result.get("_process")
    if not process:
        return
    try:
        tail = _fetch_server_log_tail(server_host, server_log_q, ssh_config, process)
        if tail:
            process["server_log_tail"] = tail
    except Exception as log_exc:
        process["server_log_tail_error"] = str(log_exc)


def _start_perf_record(
    server_pid: int | None,
    perf_record: bool,
    perf_pid_q: str,
    perf_data_q: str,
    server_host: str,
    ssh_config: dict[str, Any] | None,
    process: dict[str, Any],
) -> bool:
    """Start ``perf record -g`` on the server PID for callchain sampling.

    Returns ``True`` if perf was successfully started.
    Records the SSH command result in *process* and any error as
    ``process["perf_start_error"]``.
    """
    if not perf_record:
        return False
    if not server_pid:
        # perf was REQUESTED but the server PID could not be read (empty or
        # non-decimal pid file while the port still came up, so the run is NOT
        # aborted and its BW is valid).  Attribute the miss so the run renders
        # "n/a" in the Top-CPU-Consumers bar -- never a zero-height bar
        # masquerading as a clean "no CPU consumers" reading.  ({} stays an
        # honest zero-height bar only for perf_record=False -- perf disabled.)
        process["perf_start_error"] = "perf requested but server PID unavailable"
        return False
    perf_start = _record_command(process, "perf_start", _run_remote_result(
        f"rm -f {perf_data_q} {perf_pid_q}; "
        f"perf record -g -p {server_pid} -F 99 -o {perf_data_q} "
        f">/dev/null 2>&1 & echo $! > {perf_pid_q}",
        server_host,
        ssh_config=ssh_config,
    ))
    if not perf_start.ok:
        # perf is diagnostic overhead, not essential — log the failure
        # and proceed with the bandwidth measurement.
        process["perf_start_error"] = perf_start.error_summary()
        return False
    return True


def _read_client_time_usage(
    time_q: str,
    client_host: str,
    ssh_config: dict[str, Any] | None,
    process: dict[str, Any],
) -> dict[str, Any]:
    """Read and parse ``/usr/bin/time`` output from the client host.

    Returns a dict with keys ``client_user_sec``, ``client_sys_sec``,
    ``client_cpu_pct``, ``client_max_rss_kb`` (or empty dict on failure).
    Records the SSH command and result in *process*.
    """
    proc_usage: dict[str, Any] = {}
    time_read = _record_command(process, "client_time_read", _run_remote_result(
        f"cat {time_q} 2>/dev/null",
        client_host,
        ssh_config=ssh_config,
        sudo=False,
    ))
    if not time_read.ok:
        return {}
    time_lines = [line.strip() for line in time_read.stdout.splitlines() if line.strip()]
    if time_lines:
        parts = time_lines[-1].split()
        if len(parts) >= 4:
            proc_usage = {
                "client_user_sec": parts[0],
                "client_sys_sec": parts[1],
                "client_cpu_pct": parts[2].rstrip("%"),
                "client_max_rss_kb": parts[3],
            }
    process["client_usage"] = proc_usage
    return proc_usage


def _finalize_perf_collection(
    perf_started: bool,
    perf_pid_q: str,
    perf_data_q: str,
    server_host: str,
    ssh_config: dict[str, Any] | None,
    process: dict[str, Any],
) -> tuple[dict[str, float], str | None]:
    """Stop ``perf record``, read the report, and parse the symbols.

    Returns a ``(perf_report, error)`` tuple.

    * ``perf_report`` is the parsed perf-symbol dict (empty dict when
      *perf_started* is ``False`` or on failure).
    * ``error`` is ``None`` on success, else an error message that the
      caller should propagate as a fatal run error.
    """
    if not perf_started:
        return {}, None

    # Stop perf with SIGINT (flushes data and exits gracefully).
    # Intentionally omit rm -f here: the finally-path _cancel_perf_record already
    # handles kill-verify + pid-file removal with SIGKILL escalation and liveness
    # check.  Leaving the pid file in place ensures the finally-net can still act
    # if perf survives the SIGINT (e.g. mid-flush of a large callchain perf.data).
    perf_stop = _record_command(process, "perf_stop", _run_remote_result(
        f"ppid=$(cat {perf_pid_q} 2>/dev/null) && "
        f"kill -INT $ppid 2>/dev/null; "
        f"sleep 2",  # wait for perf to finalise perf.data
        server_host,
        timeout=15,
        ssh_config=ssh_config,
    ))
    if not perf_stop.ok:
        return {}, f"perf stop failed: {perf_stop.error_summary()}"

    perf_report_read = _record_command(process, "perf_report", _run_remote_result(
        f"perf report --stdio --no-header -g none -i {perf_data_q}",
        server_host,
        ssh_config=ssh_config,
    ))
    if not perf_report_read.ok:
        return {}, f"perf report failed: {perf_report_read.error_summary()}"

    # Best-effort temp cleanup AFTER the report was read: a failed rm -f
    # must not fail the data point, so .ok is intentionally not gated
    # here (unlike the perf_stop / perf_report checks above).
    _record_command(process, "perf_data_rm", _run_remote_result(
        f"rm -f {perf_data_q}",
        server_host,
        ssh_config=ssh_config,
    ))
    return _parse_perf_report(perf_report_read.stdout), None


def _launch_server_and_get_pid(
    tmp_dir_q: str,
    server_pid_q: str,
    server_log_q: str,
    server_cmd: str,
    run_id: str,
    server_host: str,
    ssh_config: dict[str, Any] | None,
    extra_args: list[str],
    default_port: int,
    wait_timeout: int,
    process: dict[str, Any],
) -> tuple[int | None, str | None]:
    """Launch the perftest server on the remote host and capture its PID.

    Returns ``(server_pid, error)``.  On success *server_pid* is the PID (or
    ``None`` if the PID could not be parsed) and *error* is ``None``.  On
    failure *error* is the error message.
    Records all SSH command results in *process*.
    """
    server_start = _record_command(process, "server_start", _run_remote_result(
        f"mkdir -p {tmp_dir_q}; rm -f {server_pid_q} {server_log_q}; "
        f"{server_cmd} >{server_log_q} 2>&1 & "
        f"pid=$!; "
        f"started=$(ps -p $pid -o lstart= 2>/dev/null | sed 's/^ *//'); "
        f"printf '%s\\n%s\\n%s\\n' \"$pid\" \"$started\" {shlex.quote(run_id)} > {server_pid_q}",
        server_host,
        ssh_config=ssh_config,
    ))
    if not server_start.ok:
        return None, f"server launch failed: {server_start.error_summary()}"

    port = _port_from_args(extra_args, default_port)
    _wait_for_port(server_host, port, timeout=wait_timeout, ssh_config=ssh_config)

    # Capture server PID
    server_pid_read = _record_command(process, "server_pid_read", _run_remote_result(
        f"cat {server_pid_q}",
        server_host,
        ssh_config=ssh_config,
        sudo=False,
    ))
    if not server_pid_read.ok:
        process["server_pid"] = None
        return None, f"failed to read server PID: {server_pid_read.error_summary()}"
    server_pid_raw = server_pid_read.stdout.splitlines()[0].strip() if server_pid_read.stdout.strip() else ""
    server_pid = int(server_pid_raw) if server_pid_raw and server_pid_raw.isdecimal() else None
    process["server_pid"] = server_pid
    return server_pid, None


def _read_client_result(
    json_q: str,
    client_host: str,
    ssh_config: dict[str, Any] | None,
    client_run: Any,
    binary: str,
    process: dict[str, Any],
) -> dict[str, Any]:
    """Fetch the client JSON result, parse it, and attach the *process* dict.

    Reads the ``--out_json`` output file from the remote client, parses it,
    validates metrics, and attaches the *process* tracking dict.  Returns the
    final result dict that ``run_perftest`` returns to its caller.
    """
    json_read = _record_command(process, "client_json_read", _run_remote_result(
        f"cat {json_q} 2>/dev/null",
        client_host,
        ssh_config=ssh_config,
        sudo=False,
    ))
    result = _parse_json_output(json_read.stdout)
    run_error = _check_run_errors(result, client_run, json_read, binary)
    if run_error:
        result["error"] = run_error
    result["_process"] = process
    return result


def _elevate_cleanup_errors(
    cleanup: dict[str, Any] | None,
    process: dict[str, Any],
) -> None:
    """Elevate cleanup-result errors to top-level *process* keys.

    Copies ``error`` → ``process["cleanup_error"]`` and
    ``perf_error`` → ``process["perf_cleanup_error"]`` when *cleanup* is
    non-empty, so they appear at the same prominence as a measurement error.
    The ``commands.cleanup`` dict is also recorded for forensic detail.
    """
    if not cleanup:
        return
    # NOT using _record_command: cleanup is already a plain dict, not a
    # RemoteResult — _record_command unconditionally calls .to_dict().
    process["commands"]["cleanup"] = cleanup
    cleanup_error = cleanup.get("error")
    if cleanup_error:
        process["cleanup_error"] = cleanup_error
    # A leaked perf-record process keeps sampling -- and consuming CPU --
    # on the remote host, which would skew the NEXT sweep point's
    # sys-metrics and corrupt attribution.  Surface a perf-cleanup
    # failure with the SAME prominence as a server-kill failure rather
    # than burying it one level down in commands.cleanup.perf_error.
    perf_cleanup_error = cleanup.get("perf_error")
    if perf_cleanup_error:
        process["perf_cleanup_error"] = perf_cleanup_error


def _run_client_perftest(
    tmp_dir_q: str,
    json_q: str,
    time_q: str,
    env_cmd: str,
    bin_q: str,
    server_address: str,
    args_str: str,
    duration_arg: str,
    duration: int,
    client_host: str,
    ssh_config: dict[str, Any] | None,
    process: dict[str, Any],
) -> Any:
    """Run perftest client wrapped with ``/usr/bin/time`` for resource accounting.

    Returns the SSH remote result object and records it in *process*.
    """
    time_fmt = "%U %S %P %M %c %w"
    client_run = _record_command(process, "client_run", _run_remote_result(
        f"mkdir -p {tmp_dir_q}; rm -f {json_q} {time_q}; "
        f"{env_cmd}/usr/bin/time --format={shlex.quote(time_fmt)} "
        f"{bin_q} {shlex.quote(server_address)} {args_str} {duration_arg} "
        f"--out_json --out_json_file={json_q} 2>{time_q}",
        client_host,
        timeout=int(duration) + 60,
        ssh_config=ssh_config,
    ))
    return client_run


def _make_process_tracker(
    run_id: str,
    server_host: str,
    client_host: str,
    server_address: str,
) -> dict[str, Any]:
    """Create the per-run process tracking dict used throughout ``run_perftest``."""
    return {
        "run_id": run_id,
        "server_host": server_host,
        "client_host": client_host,
        "server_address": server_address,
        "server_pid": None,
        "server_perf": {},
        "client_usage": {},
        "commands": {},
    }


def _make_server_error_result(
    error_msg: str,
    server_host: str,
    server_log_q: str,
    ssh_config: dict[str, Any] | None,
    process: dict[str, Any],
) -> dict[str, Any]:
    """Create an error result dict enriched with the perftest server log tail.

    Constructs ``{"error": error_msg, "_process": process}`` and calls
    ``_enrich_error_with_server_log`` so that every error-return path in
    ``run_perftest`` (launch failure, perf failure, exception) fetches
    the server log tail for root-cause attribution.
    """
    result: dict[str, Any] = {"error": error_msg, "_process": process}
    _enrich_error_with_server_log(result, server_host, server_log_q, ssh_config)
    return result


def _handle_run_exception(
    exc: Exception,
    server_host: str,
    server_log_q: str,
    ssh_config: dict[str, Any] | None,
    process: dict[str, Any],
) -> dict[str, Any]:
    """Handle an exception in ``run_perftest`` by recording details and fetching
    the server log tail for diagnostics.

    Delegates to ``_make_server_error_result`` so that both exception and
    non-exception error paths share the same server-log enrichment logic.

    Returns the error dict that ``run_perftest`` returns to its caller.
    """
    process["exception"] = str(exc)
    return _make_server_error_result(str(exc), server_host, server_log_q, ssh_config, process)


def run_perftest(
    binary: str,
    server_host: str,
    client_host: str,
    server_address: str,
    perftest_config: dict[str, Any],
    ssh_config: dict[str, Any],
    extra_args: list[str],
    duration: int,
    use_gpu: bool,
) -> dict[str, Any]:
    """Run one perftest measurement and return parsed JSON + metadata.

    Launches the server in the background over SSH, runs the remote client with
    ``--out_json`` (writes JSON to a file), reads the file back, then
    stops the server.  Server-side CPU attribution via ``perf record -g``,
    client-side resource accounting via ``/usr/bin/time``.
    """
    run_id = uuid.uuid4().hex[:12]
    perftest_config = resolve_perftest_paths(perftest_config, run_id=run_id)
    perftest_dir = str(perftest_config["dir"])
    bin_path = os.path.join(perftest_dir, binary)
    if use_gpu:
        bin_path += "_gpu"

    json_file = str(perftest_config["json_file"])
    time_file = str(perftest_config["time_file"])
    server_pid_file = str(perftest_config["server_pid_file"])
    server_log_file = str(perftest_config["server_log_file"])
    perf_data = str(perftest_config["perf_data"])
    perf_pid_file = str(perftest_config["perf_pid_file"])
    perf_record = parse_bool(perftest_config.get("perf_record", True), "perftest.perf_record")
    wait_timeout = _config_int(perftest_config, "wait_timeout", 30)
    default_port = _config_int(perftest_config, "default_port", 18515)
    env = _env_prefix(perftest_config)
    env_cmd = f"{env} " if env else ""

    # Build args: strip out any --out_json (we add our own).
    args_str, duration_arg = _build_perftest_cmdline(extra_args, duration)
    bin_q = shlex.quote(bin_path)
    server_pid_q = shlex.quote(server_pid_file)
    server_log_q = shlex.quote(server_log_file)
    json_q = shlex.quote(json_file)
    time_q = shlex.quote(time_file)
    perf_data_q = shlex.quote(perf_data)
    perf_pid_q = shlex.quote(perf_pid_file)
    tmp_dir_q = shlex.quote(str(perftest_config["tmp_dir"]))

    # perftest starts server mode by omitting the remote address.
    # Server must NOT receive -D (duration); perftest's server -D means
    # "exit after N seconds without a client", which races against the
    # port-probe wait below.  The server waits indefinitely by default.
    server_cmd = f"{env_cmd}{bin_q} {args_str}".strip()
    process = _make_process_tracker(run_id, server_host, client_host, server_address)

    try:
        server_pid, launch_err = _launch_server_and_get_pid(
            tmp_dir_q, server_pid_q, server_log_q, server_cmd, run_id,
            server_host, ssh_config, extra_args, default_port, wait_timeout,
            process,
        )
        if launch_err:
            return _make_server_error_result(launch_err, server_host, server_log_q, ssh_config, process)

        # Start perf record -g on server PID (callchain sampling)
        perf_started = _start_perf_record(
            server_pid, perf_record, perf_pid_q, perf_data_q,
            server_host, ssh_config, process,
        )

        # Run client wrapped with /usr/bin/time for process resource accounting.
        client_run = _run_client_perftest(
            tmp_dir_q, json_q, time_q, env_cmd, bin_q, server_address,
            args_str, duration_arg, duration, client_host, ssh_config, process,
        )

        # Parse /usr/bin/time output (client-side user/sys, RSS)
        _read_client_time_usage(time_q, client_host, ssh_config, process)

        # Stop perf, read report, parse symbols
        perf_report, perf_err = _finalize_perf_collection(
            perf_started, perf_pid_q, perf_data_q,
            server_host, ssh_config, process,
        )
        if perf_err:
            # Mirror the perf-start failure path (lines 722-726): a diagnostic
            # perf stop/report failure must NOT discard an already-valid BW
            # measurement.  Record the error and fall through to _read_client_result
            # so the BW is preserved and the perf failure is attributed separately.
            process["perf_collection_error"] = perf_err
        else:
            process["server_perf"] = perf_report

        # Read JSON result
        result = _read_client_result(
            json_q, client_host, ssh_config, client_run, binary, process,
        )
        # Fetch server log tail on client failure for root-cause diagnostics
        _enrich_error_with_server_log(result, server_host, server_log_q, ssh_config)
        return result
    except Exception as exc:
        return _handle_run_exception(exc, server_host, server_log_q, ssh_config, process)
    finally:
        cleanup = _cancel(
            server_host, ssh_config, server_pid_file, bin_path, run_id,
            perf_pid_file, perf_data,
        )
        _elevate_cleanup_errors(cleanup, process)


def _run_cleanup_cmd(
    cmd: str,
    host: str,
    ssh_config: dict[str, Any] | None,
    evidence_key: str,
    error_key: str,
) -> dict[str, Any]:
    """Run a remote cleanup command and return evidence.

    Executes *cmd* on *host* and stores the result as *evidence_key*.
    On a non-zero exit the error summary is stored as *error_key*.
    Wraps the call in try/except so a transport failure never
    propagates to the caller.

    Used by ``_cancel_perf_record`` and ``_kill_matched_server``, which
    differ only in their command strings and evidence key names.
    """
    try:
        result = _run_remote_result(cmd, host, ssh_config=ssh_config)
        evidence: dict[str, Any] = {evidence_key: result.to_dict()}
        if not result.ok:
            evidence[error_key] = result.error_summary()
        return evidence
    except Exception as exc:
        return {error_key: str(exc)}


def _cancel_perf_record(
    host: str,
    ssh_config: dict[str, Any] | None,
    perf_pid_file: str,
    perf_data: str,
) -> dict[str, Any]:
    """Safety-net cleanup for a leaked ``perf record`` process on the remote host.

    Returns an evidence dict with optional keys ``perf_cleanup`` (the SSH
    command result) and ``perf_error`` (on failure).  Returns an empty dict
    when *perf_pid_file* or *perf_data* is falsy (no-op).
    """
    if not perf_pid_file or not perf_data:
        return {}
    perf_pid_q = shlex.quote(perf_pid_file)
    perf_data_q = shlex.quote(perf_data)
    return _run_cleanup_cmd(
        f"ppid=$(cat {perf_pid_q} 2>/dev/null || true); "
        f"if [ -n \"$ppid\" ]; then "
        f"args=$(ps -p \"$ppid\" -o args= 2>/dev/null || true); "
        f"if printf '%s\\n' \"$args\" | grep -F -- {perf_data_q} >/dev/null 2>&1; then "
        f"kill -INT \"$ppid\" 2>/dev/null || true; "
        f"sleep 1; "
        f"kill -KILL \"$ppid\" 2>/dev/null || true; "
        f"if ps -p \"$ppid\" >/dev/null 2>&1; then "
        f"echo \"perf cleanup failed for pid $ppid\" >&2; exit 1; fi; "
        f"echo perf-cleaned; "
        f"else echo \"skip perf cleanup for pid $ppid: $args\" >&2; fi; "
        f"rm -f {perf_pid_q}; fi",
        host, ssh_config=ssh_config,
        evidence_key="perf_cleanup", error_key="perf_error",
    )


def _kill_matched_server(
    pid: str,
    expected_binary: str,
    expected_start: str,
    host: str,
    ssh_config: dict[str, Any] | None,
    pid_file_q: str,
) -> dict[str, Any]:
    """Kill a verified server process on the remote host.

    Three identity checks ALL gate the kill — a missing or empty
    *expected_start* means check #2 cannot pass, so the process is left
    alive as a safety guard:

      1. the pid file's run-id matches this run (checked by caller),
      2. the live process's start time (``lstart``) matches the recorded
         value (empty → kill skipped), and
      3. the live process's command line still contains the expected binary.

    Returns an evidence dict with ``cleanup`` (SSH result) on success, or
    ``cleanup`` + ``error`` on failure.
    """
    pid_q = shlex.quote(pid)
    expected_q = shlex.quote(expected_binary)
    expected_start_q = shlex.quote(expected_start)
    return _run_cleanup_cmd(
        f"args=$(ps -p {pid_q} -o args= 2>/dev/null || true); "
        f"started=$(ps -p {pid_q} -o lstart= 2>/dev/null | sed 's/^ *//'); "
        f"if [ -n {expected_start_q} ] && [ \"$started\" = {expected_start_q} ] && "
        f"printf '%s\\n' \"$args\" | grep -F -- {expected_q} >/dev/null; then "
        f"kill -TERM {pid_q} 2>/dev/null || true; "
        f"sleep 1; "
        f"kill -KILL {pid_q} 2>/dev/null || true; "
        f"if ps -p {pid_q} >/dev/null 2>&1; then "
        f"echo \"cleanup failed for pid {pid_q}\" >&2; exit 1; fi; "
        f"rm -f {pid_file_q}; echo cleaned; "
        f"else echo \"skip cleanup for pid {pid_q}: $args\" >&2; fi",
        host, ssh_config=ssh_config,
        evidence_key="cleanup", error_key="error",
    )


def _cancel(
    host: str,
    ssh_config: dict[str, Any] | None = None,
    server_pid_file: str = "",
    expected_binary: str = "",
    expected_run_id: str = "",
    perf_pid_file: str = "",
    perf_data: str = "",
) -> dict[str, Any]:
    """Kill the perftest server we launched, verifying identity before the kill.

    The kill is gated on three independent checks so we never signal an
    unrelated process that reused our recorded PID after the server exited:
      1. the pid file's run-id matches this run (our own marker, checked here),
      2. the live process's start time (``lstart``) still matches what we
         recorded at launch, and
      3. the live process's command line still contains the expected binary.
    Only when all three hold do we TERM/KILL it; afterwards we confirm the
    process is actually gone and report (via ``.ok``) if it survived.  These
    guards are deliberate — do not reduce this to an unconditional ``kill``.

    We intentionally remove only the pid file (so a stale PID can't be reused);
    the raw perftest JSON / server log under the per-run ``tmp_dir`` are left in
    place as forensic evidence — they hold detail not preserved in the parsed
    result.json, and one small dir per run is an acceptable cost for that.

    Also a safety net for ``perf record``: the happy path stops perf inline with
    SIGINT before reading its report but deliberately leaves ``perf_pid_file`` for
    this net to clear (see ``_finalize_perf_collection``).  An exception or Ctrl-C
    between perf-start and that inline stop, or a ``perf`` that outlives the SIGINT
    mid-flush of a large callchain, would otherwise leave a privileged ``perf
    record`` sampling on the server forever, per failed run.  Here we verify-kill
    it from the ``finally`` path and remove the pid file.  Idempotent: a no-op once
    ``perf_pid_file`` is gone.  PID-reuse-safe: we only signal a pid whose live
    command line still names OUR run-specific ``-o <perf_data>`` target (the
    same identity discipline as the server kill above).
    """
    evidence = _cancel_perf_record(host, ssh_config, perf_pid_file, perf_data)

    if not server_pid_file or not expected_binary or not expected_run_id:
        return evidence
    pid_file_q = shlex.quote(server_pid_file)
    try:
        pid_read = _run_remote_result(
            f"cat {pid_file_q} 2>/dev/null || true",
            host, ssh_config=ssh_config, sudo=False,
        )
        evidence["pid_read"] = pid_read.to_dict()
        pid, expected_start, pid_run_id = _parse_pid_file_lines(pid_read.stdout)
        if pid and pid.isdecimal() and pid_run_id == expected_run_id:
            evidence.update(
                _kill_matched_server(pid, expected_binary, expected_start, host, ssh_config, pid_file_q),
            )
    except Exception as exc:
        evidence.setdefault("error", str(exc))
    return evidence


def _parse_pid_file_lines(raw: str) -> tuple[str, str, str]:
    """Parse a remote PID file's 3-line content into ``(pid, expected_start, pid_run_id)``.

    The PID file has one field per line: PID, lstart timestamp, and a hex run-id.
    Missing lines produce empty strings so the caller can distinguish "file was
    empty" from "the file is not ours" (run-id mismatch).
    """
    pid_lines = raw.splitlines()
    pid = pid_lines[0].strip() if len(pid_lines) >= 1 else ""
    expected_start = pid_lines[1].strip() if len(pid_lines) >= 2 else ""
    pid_run_id = pid_lines[2].strip() if len(pid_lines) >= 3 else ""
    return pid, expected_start, pid_run_id


def _parse_json_output(raw: str | None) -> dict[str, Any]:
    if not raw or raw.strip() == "":
        return {"error": "no JSON output"}

    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
        return {"result": obj}
    except json.JSONDecodeError as exc:
        return {"error": f"JSON parse failed: {exc}", "raw_snippet": raw[:500]}


def _check_run_errors(
    result: dict[str, Any],
    client_run: Any,
    json_read: Any,
    binary: str,
) -> str:
    """Return error description if client run or JSON reading failed, else "".

    Checks, in order: failed client run (non-zero exit), missing/incomplete JSON
    output, and invalid perftest metrics.  A pre-existing ``result["error"]``
    from ``_parse_json_output`` is intentionally NOT checked here — the more
    specific ``client_run.ok`` / ``json_read.ok`` diagnostics take priority.
    """
    if not client_run.ok:
        return f"client run failed: {client_run.error_summary()}"
    if not json_read.ok:
        return f"client JSON read failed: {json_read.error_summary()}"
    return _validate_perftest_metrics(binary, result)


def _validate_perftest_metrics(binary: str, result: dict[str, Any]) -> str:
    if "error" in result:
        return ""
    metrics = result.get("results")
    if not isinstance(metrics, dict):
        return "perftest JSON missing results object"

    required = ("t_avg",) if binary.endswith("_lat") else ("BW_average", "MsgRate")
    missing = [key for key in required if metrics.get(key) is None]
    if missing:
        return f"perftest JSON missing expected field(s): {', '.join(missing)}"
    return ""


# ---------------------------------------------------------------------------
# Sweep runner
# ---------------------------------------------------------------------------

def _expand_range(param: dict[str, Any]) -> list[Any]:
    """Expand a sweep range dict (``from``/``to``/``step``) into a list of values.

    Raises ``ValueError`` on missing/non-numeric keys, non-finite (nan/inf)
    values, non-positive step, or ``to < from``.
    """
    lo = _config_val(param, "from", 0)
    # A range dict needs 'to'; a bare param["to"] would raise an opaque
    # KeyError instead of the clear, idiom-matching message below.
    if "to" not in param:
        raise ValueError(f"sweep spec needs 'values' or 'to': {param!r}")
    hi = param["to"]
    step = _config_val(param, "step", 1)
    if hi is None:
        raise ValueError(f"sweep 'to' is null in {param!r}")
    # isinstance(True, int) is True in Python — exclude bool explicitly so that
    # a YAML value like the bare word ``true`` is caught, not treated as integer 1.
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (lo, hi, step)):
        raise ValueError(
            f"sweep 'from'/'to'/'step' must be numeric (int/float; bool not accepted), "
            f"got lo={lo!r} hi={hi!r} step={step!r}"
        )
    # A nan/inf value passes the isinstance check above (both are float
    # instances) but corrupts the range arithmetic below: int(nan - lo) raises
    # an opaque ValueError, int(inf) an OverflowError, and a step of inf
    # silently collapses to a single nan value (0 * inf == nan), injecting
    # garbage into the sweep instead of failing loud.  Reject non-finite values
    # with a clear message, mirroring qp validation in _validate_qp_positive.
    if not all(math.isfinite(v) for v in (lo, hi, step)):
        raise ValueError(
            f"sweep 'from'/'to'/'step' must be finite (no nan/inf), "
            f"got lo={lo!r} hi={hi!r} step={step!r} in {param!r}"
        )
    # Fail loud on a bad sweep spec rather than hang or run zero combos: a
    # non-positive step produces a nonsensical or empty counter-based range
    # (step==0) or runs away to -inf (step<0); hi<lo silently yields an empty
    # sweep.  This code only models an ascending range, so reject anything it
    # can't represent.
    if step <= 0:
        raise ValueError(f"sweep 'step' must be positive, got {step!r} in {param!r}")
    if hi < lo:
        raise ValueError(f"sweep 'to' ({hi!r}) must be >= 'from' ({lo!r}) in {param!r}")
    # keep inclusive; use counter-based arithmetic with epsilon guard to avoid
    # float-accumulation drift (e.g. 0.1 + 0.1 + 0.1 ≈ 0.30000000000000004 > 0.3
    # would silently drop the endpoint without the epsilon).
    n_steps = int((hi - lo) / step + 1e-12)
    r = [lo + i * step for i in range(n_steps + 1)]
    return r


def _expand(param: Any) -> list[Any]:
    """Expand a parameter specification into a list of values.

    Acceptable forms:
      - scalar → [scalar]
      - list   → [...]
      - dict with 'from', 'to', 'step' → range
      - dict with 'values' → list
    """
    if isinstance(param, list):
        return param[:]
    if not isinstance(param, dict):
        return [param]

    if "values" in param:
        vals = list(param["values"])
        if not vals:
            raise ValueError(f"sweep 'values' list is empty: {param!r}")
        if any(v is None or isinstance(v, bool) for v in vals):
            raise ValueError(f"sweep 'values' list contains null or bool: {param!r}")
        return vals

    return _expand_range(param)


def _validate_qp_positive(combo: dict[str, Any]) -> None:
    """Validate that QP (queue pair count) is positive when present in *combo*.

    No-op if ``"qp"`` is not a key in *combo* (perftest defaults apply).

    QP ≤ 0 is not representable on a log2 x-axis and produces a wasted
    perftest run that will always ERR.  Catch it early with a clear message
    instead of silently filtering the point out of the chart later.

    A non-numeric ``qp`` (e.g. a YAML value quoted by mistake — ``['1', '2']``
    instead of ``[1, 2]``) is coerced for the comparison only: a numeric
    string passes through (perftest and the chart's ``float()`` both accept
    it), while a genuinely non-numeric value raises the same clear, loud
    error rather than an opaque ``TypeError`` that would abort the sweep.

    ``nan``/``inf`` (reachable from YAML ``.nan``/``.inf``) are rejected too:
    both pass a bare ``<= 0`` test, so without this guard they would slip
    through to perftest as ``-q nan`` and produce the always-ERR run this
    validator exists to catch early.
    """
    qp_val = combo.get("qp")
    if qp_val is None:
        return
    try:
        qp_num = float(qp_val)
    except (TypeError, ValueError):
        raise ValueError(
            f"QP must be numeric, got {qp_val!r}. "
            "Use unquoted numbers in the QP values list (e.g. [1, 2, 4], not ['1', '2'])."
        ) from None
    if not math.isfinite(qp_num):
        raise ValueError(
            f"QP must be a finite number, got {qp_val!r}. "
            "Remove nan/inf from the QP values list."
        )
    if qp_num <= 0:
        raise ValueError(
            f"QP must be positive, got {qp_val}. "
            "Remove 0 from the QP values list, or set 'from: 1' if using a range spec."
        )


def sweep_config(config: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Yield every parameter combination from the config's sweep spec.

    This function consumes only two TOP-LEVEL config keys: ``sweep`` (the list
    of swept parameters) and ``fixed`` (parameters applied to every combo).
    The rest of the config (``test``, ``duration``, ``use_gpu``,
    ``server_host``/``server_address``/``client_host``, ``perftest``, ``ssh``,
    ``report``, ...) is read separately by ``runtime_config`` and is NOT nested
    under ``sweep``.

    Relevant structure::

        fixed:
          port: 18515                    # applied to every combination
        sweep:
          - name: msg_size
            values: [1, 4, 64, 1024]     # explicit byte sizes
          - name: qp
            from: 1
            to: 512
            step: 64                     # linear range; step must be > 0
          # for a non-linear (e.g. geometric) sweep, list the points via `values`:
          #   - name: qp
          #     values: [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    """
    sweep_params = _config_val(config, "sweep", [])
    fixed = _config_val(config, "fixed", {})

    expanded: list[tuple[str, list[Any]]] = []
    for sp in sweep_params:
        if not isinstance(sp, dict) or "name" not in sp:
            raise ValueError(f"sweep entry missing 'name': {sp!r}")
        name = sp["name"]
        expanded.append((name, _expand(sp)))

    keys = [e[0] for e in expanded]
    for values in itertools.product(*[e[1] for e in expanded]):
        combo: dict[str, Any] = {}
        combo.update(fixed)
        for k, v in zip(keys, values):
            combo[k] = v
        _validate_qp_positive(combo)
        yield combo


BUILD_ARGS_FLAG_MAP: dict[str, str] = {
    "msg_size": "-s",
    "qp": "-q",
    "tx_depth": "-t",
    "rx_depth": "-r",
    "post_list": "-l",
    "cq_mod": "-Q",
    "iters": "-n",
    "port": "-p",
    "ib_port": "-i",
    "inline": "-I",
    "sl": "-S",
    "mtu": "-m",
    "tos": "-T",
    "recv_post_list": "--recv-post-list",
    "cpu_util": "--cpu_util",
    "device": "-d",
    "gid_index": "-x",
    "force_link": "--force-link",
    "rdma_cm": "-R",
    "comm_rdma_cm": "-z",
    "bind_source_ip": "--bind_source_ip",
    "check_alive": "--check-alive",
}

BUILD_ARGS_SKIP_KEYS: frozenset[str] = frozenset({
    "client",
    "client_host",
    "duration",
    "host",
    "output_dir",
    "perftest",
    "perftest_dir",
    "rdma_core_lib",
    "report",
    "server",
    "server_addr",
    "server_address",
    "server_host",
    "ssh",
    "test",
    "use_gpu",
})


def _build_args(combo: dict[str, Any]) -> list[str]:
    """Convert a parameter combo into perftest CLI arguments."""
    args: list[str] = []
    for k, v in combo.items():
        if k in BUILD_ARGS_SKIP_KEYS:
            continue
        flag = BUILD_ARGS_FLAG_MAP.get(k, f"--{k.replace('_', '-')}")
        # Fail loud on a key written with no usable value (YAML `inline:` ->
        # None, `inline: ""`, or an empty `inline: []`/`{}`).  Forwarding it
        # would emit the literal ["-I", "None"]/["-I", "[]"]; perftest then
        # exits non-zero and the run renders "ERR" honestly -- but the failure
        # surfaces far from its cause.  A valueless toggle is an explicit bool
        # (handled below), never None; a real 0 is a value and passes through.
        has_no_value = (
            v is None
            or (isinstance(v, str) and not v.strip())
            or (isinstance(v, (list, dict)) and not v)
        )
        if has_no_value:
            raise ValueError(
                f"config key {k!r} has no value (got {v!r}); "
                "remove it or give it a value"
            )
        if isinstance(v, bool):
            if v:
                args.append(flag)
        else:
            args.append(flag)
            args.append(str(v))
    return args


def _build_run_metadata(
    test: str,
    server: dict[str, Any],
    client: dict[str, Any],
    duration: int,
    combo: dict[str, Any],
    elapsed: float,
    server_cpu_diff: dict[str, Any],
    client_cpu_diff: dict[str, Any],
    server_mem_info: dict[str, Any],
    client_mem_info: dict[str, Any],
    server_before: dict[str, Any],
    server_after: dict[str, Any],
    client_before: dict[str, Any],
    client_after: dict[str, Any],
) -> dict[str, Any]:
    """Build the ``_meta`` dict attached to every per-combo result."""
    return {
        "timestamp": timestamp(),
        "test": test,
        "server": server,
        "client": client,
        "duration": duration,
        "parameters": combo,
        "elapsed_sec": round(elapsed, 2),
        # Unprefixed keys ("cpu_util_per_core"/"memory") are backward-compat
        # aliases for the SERVER metrics, kept so older report consumers and
        # pre-existing result files keep working; new code reads "server_*".
        "cpu_util_per_core": server_cpu_diff,
        "server_cpu_util_per_core": server_cpu_diff,
        "client_cpu_util_per_core": client_cpu_diff,
        "memory": server_mem_info,
        "server_memory": server_mem_info,
        "client_memory": client_mem_info,
        "server_softirqs": SysMonitor.compute_softirq_diff(server_before, server_after),
        "client_softirqs": SysMonitor.compute_softirq_diff(client_before, client_after),
        "server_sys_before_ok": "error" not in server_before,
        "server_sys_after_ok": "error" not in server_after,
        "client_sys_before_ok": "error" not in client_before,
        "client_sys_after_ok": "error" not in client_after,
    }


def _save_combo_result(
    result: dict[str, Any],
    meta: dict[str, Any],
    out_path: Path,
    combo_idx: int,
    elapsed: float,
    result_files: list[Path],
) -> Path:
    """Annotate *result* with *meta*, write per-combo result JSON to disk.

    Returns the path to the written JSON file and appends it to *result_files*.
    """
    result["_meta"] = meta
    if "error" in result:
        result["_meta"]["run_error"] = result["error"]
    else:
        # Promote cleanup/perf-cleanup errors so they appear in the summary error
        # column, are counted by _count_summary_errors, and cause main() to exit
        # non-zero.  A leaked perf-record sampler (perf_cleanup_error) is flagged
        # with the same prominence since it contaminates the NEXT combo's attribution.
        process_info = result.get("_process", {})
        cleanup_err = process_info.get("cleanup_error") or process_info.get("perf_cleanup_error")
        if cleanup_err:
            result["_meta"]["run_error"] = f"cleanup_error: {cleanup_err}"

    combo_dir = out_path / f"{combo_idx:04d}"
    combo_dir.mkdir(parents=True, exist_ok=True)
    combo_file = combo_dir / "result.json"
    _write_json(combo_file, result)
    result_files.append(combo_file)
    print(f"  → {combo_file}  ({elapsed:.1f}s)", flush=True)
    return combo_file


def _write_summary(result_files: list[Path], out_path: Path) -> None:
    """Write master ``summary.json`` and ``summary.csv`` from per-combo results."""
    summary: list[dict[str, Any]] = []
    for f in sorted(result_files):
        data = _read_json(f)
        summary.append(_summary_entry(data.get("_meta") or {}, data.get("results") or {}))

    summary_file = out_path / "summary.json"
    _write_json(summary_file, summary)
    print(f"\nSummary → {summary_file}")

    csv_path = out_path / "summary.csv"
    _write_csv(csv_path, summary)
    print(f"CSV     → {csv_path}")


def _compute_node_deltas(
    monitor: "SysMonitor",
    before: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, float], dict[str, Any]]:
    """Grab post-run system state and compute CPU/memory deltas for one node.

    Returns *(after, cpu_diff, mem_info)*.  Shared by server and client in
    ``_compute_sys_deltas`` to avoid duplicating the grab/diff/extract chain.
    """
    after = monitor.grab()
    cpu_diff = SysMonitor.compute_cpu_diff(before, after)
    mem_info: dict[str, Any] = {
        **SysMonitor.extract_mem(after),
        **SysMonitor.compute_mem_delta(before, after),
    }
    return after, cpu_diff, mem_info


def _compute_sys_deltas(
    server_monitor: "SysMonitor",
    client_monitor: "SysMonitor",
    server_before: dict[str, Any],
    client_before: dict[str, Any],
) -> tuple[
    dict[str, Any], dict[str, Any],
    dict[str, float], dict[str, float],
    dict[str, Any], dict[str, Any],
]:
    """Grab post-test system state and compute CPU/memory deltas for both nodes.

    Returns *server_after*, *client_after*, *server_cpu_diff*,
    *client_cpu_diff*, *server_mem_info*, *client_mem_info*.
    """
    server_after, server_cpu_diff, server_mem_info = _compute_node_deltas(
        server_monitor, server_before,
    )
    client_after, client_cpu_diff, client_mem_info = _compute_node_deltas(
        client_monitor, client_before,
    )
    return server_after, client_after, server_cpu_diff, client_cpu_diff, server_mem_info, client_mem_info


def _run_one_combo(
    combo_idx: int,
    combo: dict[str, Any],
    runtime: dict[str, Any],
    server_monitor: SysMonitor,
    client_monitor: SysMonitor,
    out_path: Path,
    result_files: list[Path],
) -> Path:
    """Execute one sweep combo: build args, run perftest, capture sys deltas, save."""
    test = runtime["test"]
    server_host = runtime["server"]["host"]
    client_host = runtime["client"]["host"]
    server_address = runtime["server"]["address"]
    perftest_config = runtime["perftest"]
    ssh_config = runtime["ssh"]
    default_duration = runtime["duration"]
    use_gpu = runtime["use_gpu"]

    extra_args = _build_args(combo)
    extra_args_str = " ".join(extra_args)
    duration = _config_int(combo, "duration", default_duration)
    label = f"[{combo_idx}] {test} msg={combo.get('msg_size','?')} qp={combo.get('qp','?')}"

    print(
        f"{timestamp()}  {label}  {client_host} -> {server_address}  args: {extra_args_str}",
        flush=True,
    )

    # Sample sys state before test
    server_before = server_monitor.grab()
    client_before = client_monitor.grab()

    t0 = time.monotonic()
    result = run_perftest(
        binary=test,
        server_host=server_host,
        client_host=client_host,
        server_address=server_address,
        perftest_config=perftest_config,
        ssh_config=ssh_config,
        extra_args=extra_args,
        duration=duration,
        use_gpu=use_gpu,
    )
    elapsed = time.monotonic() - t0

    (
        server_after, client_after, server_cpu_diff, client_cpu_diff,
        server_mem_info, client_mem_info,
    ) = _compute_sys_deltas(
        server_monitor, client_monitor, server_before, client_before,
    )

    # Annotate with metadata
    meta = _build_run_metadata(
        test,
        runtime["server"],
        runtime["client"],
        duration,
        combo,
        elapsed,
        server_cpu_diff,
        client_cpu_diff,
        server_mem_info,
        client_mem_info,
        server_before,
        server_after,
        client_before,
        client_after,
    )
    return _save_combo_result(result, meta, out_path, combo_idx, elapsed, result_files)


def run_sweep(config: dict[str, Any], output_dir: str = "sweep_results") -> list[Path]:
    """Execute the full sweep defined by *config*.

    Returns paths to per-combination JSON result files.
    """
    runtime = _runtime_config(config)
    server_host = runtime["server"]["host"]
    client_host = runtime["client"]["host"]
    ssh_config = runtime["ssh"]

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    _write_json(out_path / "run_config.json", runtime)

    server_monitor = SysMonitor(server_host, ssh_config)
    client_monitor = SysMonitor(client_host, ssh_config)

    result_files: list[Path] = []

    # Eagerly materialise combos so that a validation failure (e.g. non-positive
    # QP) raises before any perftest run is dispatched, not mid-sweep with
    # orphaned runs already launched.
    _combos = list(sweep_config(config))
    try:
        for combo_idx, combo in enumerate(_combos, 1):
            _run_one_combo(combo_idx, combo, runtime, server_monitor, client_monitor, out_path, result_files)
    finally:
        # Write summary even on a mid-sweep abort (disk error, scheduler kill, etc.)
        # so completed combos remain reportable without manual reconstruction.
        if result_files:
            _write_summary(result_files, out_path)
    return result_files


PERFTEST_METRIC_KEYS: tuple[str, ...] = (
    "BW_average", "MsgRate", "BW_peak", "n_iterations", "MsgSize", "t_avg",
)


def _copy_perftest_metrics(results: dict[str, Any] | None, entry: dict[str, Any]) -> None:
    """Copy perftest headline metrics to the entry's top level.

    ``t_avg`` is the latency tests' headline metric — the only latency key
    the schema validates (see ``_validate_perftest_metrics``) — and was
    otherwise collected then dropped, so a successful ``*_lat`` sweep
    produced a report with no latency number at all.  This ensures it
    appears in ``summary.json`` and ``summary.csv``.
    """
    if isinstance(results, dict):
        for pk in PERFTEST_METRIC_KEYS:
            if results.get(pk) is not None:
                entry[pk] = results[pk]


def _sys_ok(meta: dict[str, Any], side: str) -> bool:
    """AND of before+after system-monitor grab success flags for one node.

    A failed /proc grab makes every cpu/mem diff empty, which is otherwise
    indistinguishable from a genuinely idle host.  This boolean lets the
    CSV/report flag a failed measurement instead of presenting it as a clean
    low-load result.  Unknown (pre-flag result files) defaults to ``True``
    to preserve old behaviour rather than invent a failure.
    """
    return bool(meta.get(f"{side}_sys_before_ok", True)) and bool(meta.get(f"{side}_sys_after_ok", True))


def _summary_entry(meta: dict[str, Any], results: dict[str, Any] | None) -> dict[str, Any]:
    """Build one summary row from a result file's ``_meta`` and ``results``.

    Propagates the sys-monitor success flags via ``_sys_ok`` (see there for
    the masquerade-trust rationale), precomputes ``server_cpu_avg``/
    ``client_cpu_avg`` from per-core dicts, and copies perftest headline
    metrics (BW_average, MsgRate, t_avg, etc.) to the top level so they
    appear in summary.json and summary.csv.
    """
    entry: dict[str, Any] = {
        "server": _get_dict(meta, "server"),
        "client": _get_dict(meta, "client"),
        "params": _get_dict(meta, "parameters"),
        "error": meta.get("run_error"),
        "elapsed_sec": meta.get("elapsed_sec"),
        "cpu_per_core": _get_dict(meta, "cpu_util_per_core"),
        "server_cpu_per_core": _get_dict(meta, "server_cpu_util_per_core"),
        "client_cpu_per_core": _get_dict(meta, "client_cpu_util_per_core"),
        "memory": _get_dict(meta, "memory"),
        "server_memory": _get_dict(meta, "server_memory"),
        "client_memory": _get_dict(meta, "client_memory"),
        "server_sys_ok": _sys_ok(meta, "server"),
        "client_sys_ok": _sys_ok(meta, "client"),
    }
    entry["server_cpu_avg"] = _cpu_avg(entry["server_cpu_per_core"])
    entry["client_cpu_avg"] = _cpu_avg(entry["client_cpu_per_core"])
    _copy_perftest_metrics(results, entry)
    return entry


def _or_err(value: Any, is_ok: bool) -> str:
    """Render *value* or ``"ERR"`` — masquerade-trust guard for failed measurements.

    A failed /proc grab (or a perftest run that exited non-zero) makes every
    CPU/memory delta empty, which is otherwise indistinguishable from a
    genuinely idle host.  All cells in such a row are rendered ``"ERR"`` so a
    failed measurement is never read as a clean low-load result by either a
    human or a programmatic consumer.

    See also ``_csv_data_row`` and the ``server_sys_ok``/``client_sys_ok`` flags.
    """
    return str(value) if is_ok else "ERR"


def _mem_cell_pair(mem: dict[str, Any], sys_ok: bool) -> tuple[str, str]:
    """Extract ``MemUsed`` and ``MemUsedDelta`` as ``_or_err``-guarded cells.

    The two values always travel together — both the server and client memory
    dict have the same shape — so a single call replaces two manual
    ``_or_err(mem.get(...), sys_ok)`` lines, keeping the pair in sync.
    """
    return (
        _or_err(mem.get("MemUsed", ""), sys_ok),
        _or_err(mem.get("MemUsedDelta", ""), sys_ok),
    )


def _csv_data_row(
    entry: dict[str, Any],
    param_keys_sorted: list[str],
    perf_keys: list[str],
) -> list[str]:
    """Build one CSV data row from a summary entry."""
    params = _get_dict(entry, "params")
    server = _get_dict(entry, "server")
    client = _get_dict(entry, "client")
    server_cpu_avg = _cpu_avg(entry.get("server_cpu_per_core") or entry.get("cpu_per_core") or {})
    client_cpu_avg = _cpu_avg(entry.get("client_cpu_per_core") or {})
    server_mem = _get_dict(entry, "server_memory", "memory")
    client_mem = _get_dict(entry, "client_memory")
    # When a /proc grab failed, every diff is empty — indistinguishable from a
    # genuinely idle host.  Render those cells "ERR" (not blank/0) so a failed
    # measurement can never be read as a clean low-load result, and expose the
    # raw flag in dedicated columns for programmatic consumers.
    server_sys_ok = entry.get("server_sys_ok", True)
    client_sys_ok = entry.get("client_sys_ok", True)
    server_cpu_cell = _or_err(server_cpu_avg, server_sys_ok)
    client_cpu_cell = _or_err(client_cpu_avg, client_sys_ok)
    server_mem_used, server_mem_delta = _mem_cell_pair(server_mem, server_sys_ok)
    client_mem_used, client_mem_delta = _mem_cell_pair(client_mem, client_sys_ok)
    row: list[str] = [
        str(server.get("host", "")),
        str(server.get("address", "")),
        str(client.get("host", "")),
    ]
    row += [str(params.get(k, "")) for k in param_keys_sorted]
    # A run that errored can still carry a stale BW parsed from the partial
    # JSON the client wrote before exiting non-zero.  Render those cells
    # "ERR" so a failed measurement is never read as a clean bandwidth
    # number (mirrors the *_sys_ok handling above for the sys-monitor path).
    if entry.get("error"):
        row += ["ERR" for _ in perf_keys]
    else:
        row += [str(entry[k]) if entry.get(k) is not None else "" for k in perf_keys]
    row += [
        str(entry.get("error", "") or ""),
        str(entry["elapsed_sec"]) if entry.get("elapsed_sec") is not None else "",
        server_cpu_cell,
        client_cpu_cell,
        server_cpu_cell,
        str(server_sys_ok),
        str(client_sys_ok),
        server_mem_used,
        server_mem_delta,
        client_mem_used,
        client_mem_delta,
    ]
    return row


def _write_csv(path: Path, summary: list[dict[str, Any]]) -> None:
    if not summary:
        return

    # Collect all keys from params and top-level
    param_keys: set[str] = set()
    for entry in summary:
        param_keys.update(_get_dict(entry, "params").keys())
    param_keys_sorted = sorted(k for k in param_keys if k != "report_json")

    rows: list[list[str]] = []
    # Perftest metric columns: bandwidth/message-rate for *_bw tests and t_avg
    # for *_lat tests.  All are gated on entry["error"] below so a failed run
    # never renders a stale number.  Sourced from PERFTEST_METRIC_KEYS (the
    # same set _copy_perftest_metrics writes into each summary entry) so the
    # CSV columns and the summary fields can never silently drift apart.
    perf_keys = list(PERFTEST_METRIC_KEYS)
    # header
    headers = [
        "server_host",
        "server_address",
        "client_host",
        *param_keys_sorted,
        *perf_keys,
        "error",
        "elapsed_sec",
        "server_cpu_avg",
        "client_cpu_avg",
        "cpu_avg",         # backward-compat alias: identical to server_cpu_avg
        "server_sys_ok",
        "client_sys_ok",
        "server_mem_used",
        "server_mem_used_delta",
        "client_mem_used",
        "client_mem_used_delta",
    ]
    rows.append(headers)

    for entry in summary:
        rows.append(_csv_data_row(entry, param_keys_sorted, perf_keys))

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerows(rows)


def _cpu_core_keys(cpu_data: dict[str, Any]) -> list[str]:
    """Filter dict keys to per-core IDs (``cpu0``, ``cpu1``, ...).

    Excludes the aggregate ``cpu`` key from ``/proc/stat`` (which sums all
    cores) so averaging and table-building operate on individual cores only.
    """
    return [k for k in cpu_data if k.startswith("cpu") and k != "cpu"]


def _cpu_avg(cpu_per_core: dict[str, Any]) -> float | str:
    """Average the per-core CPU values, skipping cores with no reading.

    ``_cpu_core_keys`` filters by key *name*, never by value, so a corrupt or
    partially-written ``result.json`` carrying ``{"cpu0": null}`` would reach
    ``float(None)`` and abort the entire report (the same present-``None``-from
    -disk threat ``_extract_metric`` documents).  Skip ``None`` cores so a
    missing core degrades like the per-core table's "n/a" instead of crashing,
    and so the average stays honest over the cores that *were* measured rather
    than folding a missing reading in as a fabricated 0% (``float(v or 0)``
    would masquerade an absent core as a clean idle one).  An all-``None`` dict
    yields ``""``, matching the empty-dict rendering.
    """
    cpu_vals = [
        float(cpu_per_core[k])
        for k in _cpu_core_keys(cpu_per_core)
        if cpu_per_core[k] is not None
    ]
    return round(sum(cpu_vals) / len(cpu_vals), 1) if cpu_vals else ""


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _mem_delta_mib(entry: dict[str, Any], key: str) -> float:
    memory = _get_dict(entry, key)
    raw = memory.get("MemUsedDelta", "")
    try:
        return parse_size(str(raw)) / (1024 * 1024)
    except (TypeError, ValueError, OverflowError):
        # OverflowError covers an inf-shaped delta string ("infT" -> int(inf))
        # in a corrupt/hand-edited result.json; it is not a ValueError subclass,
        # so it would otherwise escape this guard and abort the whole report.
        # Return 0.0 like every other malformed MemUsedDelta -- the sys_ok flags
        # already drop failed-grab points from the plotted series.
        return 0.0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

SVG_COLORS = [
    "#2563eb", "#dc2626", "#16a34a", "#ca8a04",
    "#9333ea", "#0891b2", "#be123c", "#d1d5db",
]


class MetricSeries(NamedTuple):
    """Composite return value from _compute_metric_series."""

    qps: list[int | float]
    ok_idx: list[int]
    bw_x: list[float]
    bw_vals: list[float]
    rate_vals: list[float]
    lat_vals: list[float]
    is_latency: bool


def _extract_metric(
    summary: list[dict[str, Any]],
    ok_idx: list[int],
    key: str,
    default: int | float = 0,
) -> list[float]:
    """Extract a metric value from summary entries, guarding against ``None``.

    ``dict.get(key, default)`` returns *default* only when *key* is **absent**,
    not when the stored value is ``None`` (YAML ``null`` in ``result.json``).
    The ``or default`` guard coerces stored-``None`` to *default* so the
    ``float()`` call does not crash with ``TypeError``.  Genuine ``0.0``
    measurements are valid per ``_validate_perftest_metrics`` and pass through
    unchanged (``0.0 or 0 → 0.0``).
    """
    return [float(summary[i].get(key, default) or default) for i in ok_idx]


def _compute_metric_series(
    summary: list[dict[str, Any]],
) -> MetricSeries:
    """Extract core metric data from summary.

    Compute ok_idx (successful-run indices), extract BW/rate/latency values
    over ok_idx, and classify the sweep as latency vs throughput using a
    fallback-scope heuristic for all-failed runs.
    """
    qps = [
        i + 1 if (v := _get_dict(e, "params").get("qp")) is None else v
        for i, e in enumerate(summary)
    ]
    ok_idx = [i for i, e in enumerate(summary) if not e.get("error")]
    bw_x = [float(qps[i]) for i in ok_idx]
    bw_vals = _extract_metric(summary, ok_idx, "BW_average")
    rate_vals = _extract_metric(summary, ok_idx, "MsgRate")

    # ib_*_lat sweeps carry t_avg (us) and NO BW_average/MsgRate.  Plotting the
    # throughput panels regardless would render a valid latency run as a flat-zero
    # bandwidth curve with the latency number shown NOWHERE — a valid measurement
    # displayed as absent (the inverse of the failed-run masquerade, in the SVG,
    # the tool's headline output).  Decide which metric to plot from what the
    # SUCCESSFUL runs actually measured.  Only the successful runs are PLOTTED
    # (lat_vals over ok_idx), but classify the metric family over a wider scope:
    # if EVERY run errored (ok_idx empty) the errored entries may still carry
    # t_avg, so fall back to the whole summary — otherwise an all-failed *_lat
    # sweep would be mis-titled "Bandwidth ... no valid runs" instead of the
    # honest "Latency ... no valid runs".  If no run of either family produced a
    # metric at all, it is genuinely unclassifiable and defaults to throughput.
    lat_vals = _extract_metric(summary, ok_idx, "t_avg")
    scope = ok_idx or list(range(len(summary)))
    has_bw = any("BW_average" in summary[i] for i in scope)
    is_latency = not has_bw and any("t_avg" in summary[i] for i in scope)
    return MetricSeries(qps, ok_idx, bw_x, bw_vals, rate_vals, lat_vals, is_latency)


def _load_perf_bar_series(summary: list[dict[str, Any]], ok_idx: list[int]) -> tuple[list[dict[str, Any] | None], list[str], list[tuple[str, list[float]]]]:
    """Build stacked-bar chart series from per-run server_perf profiles."""
    perf_data: list[dict[str, Any] | None] = []
    for i in ok_idx:
        p = Path(summary[i].get("_result_path", f"{i+1:04d}/result.json"))
        if p.exists():
            try:
                d = _read_json(p)
                proc = _get_dict(d, "_process")
                if proc.get("perf_start_error") or proc.get("perf_collection_error"):
                    # ponytail: a profile was ATTEMPTED but its start or stop/report
                    # failed, leaving no server_perf.  None → "n/a" marker: never a
                    # zero-height bar masquerading as a clean "no CPU consumers"
                    # reading.  ({} below stays zero-height — perf disabled / a real
                    # profile that genuinely found nothing, both honestly empty.)
                    perf_data.append(None)
                else:
                    perf_data.append(_get_dict(proc, "server_perf"))
            except (json.JSONDecodeError, OSError):
                # ponytail: None sentinel = profile unavailable (corrupt/missing file),
                # distinct from {} = profile present but empty (perf disabled).
                perf_data.append(None)
        else:
            perf_data.append(None)

    all_syms = set()
    for pd in perf_data:
        if pd is None:
            continue
        for s, v in sorted(pd.items(), key=lambda x: -(x[1] or 0))[:5]:
            # ``v or 0`` mirrors the None-safe sort key above: a corrupt or
            # hand-edited result.json could carry a null self-% here, and a
            # bare ``None > 0`` would raise TypeError and abort the whole
            # report instead of just skipping the bad symbol.
            if (v or 0) > 0:
                all_syms.add(s)
    sym_total = {s: sum(pd.get(s, 0) or 0 for pd in perf_data if pd is not None) for s in all_syms}
    top_syms = sorted(sym_total, key=lambda s: -sym_total[s])[:7]

    # None entries (unavailable profile) get 0 for every symbol — the bar column
    # will render with zero height; _stacked_bar receives the None list so it can
    # annotate those columns with "n/a".
    series = [(sym, [pd.get(sym, 0) or 0 if pd is not None else 0 for pd in perf_data]) for sym in top_syms]
    return perf_data, top_syms, series


def _compute_sys_resource_data(summary: list[dict[str, Any]], qps: list[int | float]) -> tuple[list[float], list[float], list[float], list[float], list[float], list[bool], list[bool], str]:
    """Extract CPU/memory/resource data from summary for SVG sys-resource panels."""
    cpu_key = "server_cpu_per_core" if any("server_cpu_per_core" in e for e in summary) else "cpu_per_core"
    server_cpu_vals = [
        _as_float(e.get("server_cpu_avg", _cpu_avg(e.get(cpu_key) or {})))
        for e in summary
    ]
    client_cpu_vals = [
        _as_float(e.get("client_cpu_avg", _cpu_avg(e.get("client_cpu_per_core") or {})))
        for e in summary
    ]
    server_mem_vals = [_mem_delta_mib(e, "server_memory") for e in summary]
    client_mem_vals = [_mem_delta_mib(e, "client_memory") for e in summary]
    # A failed /proc grab leaves every cpu/mem diff empty, which _as_float() /
    # _mem_delta_mib() coerce to 0.0 — on a chart that is indistinguishable from
    # a genuinely idle host.  Carry the same server/client_sys_ok flags the CSV
    # uses and drop those points from the perf-host series (and render the table
    # cells "ERR") so a failed measurement is never plotted as a clean low-load
    # line.  Surviving points keep their real qp on the shared x-axis.
    qf = [float(q) for q in qps]
    server_ok = [bool(e.get("server_sys_ok", True)) for e in summary]
    client_ok = [bool(e.get("client_sys_ok", True)) for e in summary]
    return server_cpu_vals, client_cpu_vals, server_mem_vals, client_mem_vals, qf, server_ok, client_ok, cpu_key


def _core_sort_key(c: str) -> int:
    """Sort key for ``cpu<N>`` strings; non-standard keys sort before ``cpu0``."""
    try:
        return int(c.replace("cpu", ""))
    except ValueError:
        return -1


def _core_cell(per_core: dict[str, Any], core: str) -> str:
    # ``cores`` is the UNION across all runs, so a successful run can lack a
    # core that another run measured (a cpu offline / hot-plugged mid-sweep).
    # Render that gap as "n/a" — NOT a fabricated 0.0% idle reading — so an
    # absent measurement is never mistaken for a clean one (the inverse of
    # the masquerade we guard).  "ERR" below is the distinct case where the
    # whole /proc grab for that run failed.
    #
    # ``.get(core)`` with an ``is not None`` test (not ``core in``) also folds a
    # present-but-``null`` value from a corrupt/partial result.json into the
    # "n/a" case: the key exists but holds no reading, so ``f"{None:.1f}"`` would
    # otherwise crash the whole report.  This is the per-core *table* sibling of
    # the same present-None-from-disk guard in ``_cpu_avg`` / ``_extract_metric``
    # — ``_svg_chart`` consumes this dict on both paths in one report build.  A
    # genuine idle ``0.0`` is *not* ``None``, so it still renders "0.0", never
    # "n/a" (a real measurement must never read as absent).
    v = per_core.get(core)
    return f"{v:.1f}" if v is not None else "n/a"


def _compute_core_table_data(
    summary: list[dict[str, Any]], cpu_key: str, qps: list[int | float], server_ok: list[bool]
) -> tuple[list[str], list[list[str]]]:
    # Derive the per-core column set from the UNION of all runs, not summary[0]:
    # if the first run's /proc grab failed, its per-core dict is empty, which would
    # erase the whole table — dropping every later run's valid per-core data (a
    # valid measurement rendered as absent, the reverse of the masquerade we guard).
    core_set: set[str] = set()
    for e in summary:
        core_set.update(_cpu_core_keys(e.get(cpu_key) or {}))
    cores = sorted(core_set, key=lambda c: _core_sort_key(c))
    table_hdrs = ["QP"] + cores

    table_rows = [
        [str(q)] + (
            [_core_cell(summary[i].get(cpu_key) or {}, c) for c in cores]
            if server_ok[i] else ["ERR" for _ in cores]
        )
        for i, q in enumerate(qps)
    ]

    return table_hdrs, table_rows


def _chart_title(
    summary: list[dict[str, Any]],
    report_config: dict[str, Any] | None,
) -> tuple[str, str]:
    """Determine the SVG report title and subtitle from config or data."""
    if report_config:
        title = _str_field(report_config, "title", "RDMA Perftest Sweep")
        subtitle = _str_field(report_config, "subtitle")
    else:
        first = summary[0]
        server = _get_dict(first, "server")
        client = _get_dict(first, "client")
        title = "RDMA Perftest Sweep"
        subtitle = (
            f"{client.get('host', 'client')} -> "
            f"{server.get('address', server.get('host', 'server'))}"
        )
    return title, subtitle


def _chart_sys_xy(
    vals: list[float],
    mask: list[bool],
    qf: list[float],
) -> tuple[list[float], list[float]]:
    """Filter *vals* and *qf* by *mask* for SVG chart series."""
    xs = [q for q, m in zip(qf, mask) if m]
    ys = [v for v, m in zip(vals, mask) if m]
    return xs, ys


def _chart_sys_series(
    name: str,
    vals: list[float],
    mask: list[bool],
    qf: list[float],
    color: str,
) -> tuple[str, list[float], list[float], str]:
    """Build a named chart series tuple from system resource data.

    Wraps ``_chart_sys_xy`` and returns a ``(name, xs, ys, color)`` tuple
    ready for use in ``_multi_line_chart`` series definitions.  Used by
    ``_chart_mid_panel_defs`` for the CPU and memory pressure panels.
    """
    xs, ys = _chart_sys_xy(vals, mask, qf)
    return (name, xs, ys, color)


def _draw_y_grid(
    el: list[str],
    cx: float, cy: float, cw: float, ch: float,
    labels: list[str],
) -> None:
    """Draw 5 horizontal grid lines with right-aligned y-axis labels plus the
    baseline axis.

    *labels* holds the 5 y-axis tick strings (top → bottom).  Shared by
    ``_draw_chart_grid`` (data-scaled via ``_fmt``) and ``_stacked_bar``
    (fixed 0-100% scale).
    """
    for i in range(5):
        gy = cy + ch * i / 4
        el.append(f"<line x1='{cx}' y1='{gy}' x2='{cx + cw}' y2='{gy}' stroke='#e2e8f0' stroke-width='1'/>")
        el.append(f"<text x='{cx - 6}' y='{gy + 4}' font-family='system-ui,sans-serif' font-size='10' fill='#64748b' text-anchor='end'>{labels[i]}</text>")
    el.append(f"<line x1='{cx}' y1='{cy + ch}' x2='{cx + cw}' y2='{cy + ch}' stroke='#94a3b8' stroke-width='1'/>")


def _chart_top_panel_defs(
    is_latency: bool,
    lat_vals: list[float],
    bw_vals: list[float],
    rate_vals: list[float],
) -> list[tuple[str, str, list[float], str, int]]:
    """Build top-panel definitions for the SVG chart.

    Returns a list of ``(label, y_label, values, color, decimal_places)``
    tuples — one per throughput/latency panel.
    """
    # dec = decimal places for the chart's value labels: 0 for throughput
    # (hundreds/thousands), 2 for latency (single-digit us would otherwise round
    # to a meaningless integer — a fresh soft-masquerade we must not introduce).
    if is_latency:
        return [("Latency t_avg (us)", "us", lat_vals, "#7c3aed", 2)]
    return [
        ("Bandwidth (MB/s)", "MB/s", bw_vals, "#2563eb", 0),
        ("Message Rate (Kmsg/s)", "Kmsg/s", [r * 1000 for r in rate_vals], "#16a34a", 0),
    ]


def _chart_latency_na_panel(el: list[str], M: float, y: float, W: float,
                            h: float) -> None:
    """Render a placeholder "N/A for latency test" panel in the chart's
    second top-row slot (which would normally hold the throughput panel).

    *h* is the panel height — passed (rather than hardcoded) so the N/A
    placeholder always matches the vertical footprint of the throughput
    column it replaces, regardless of any future height adjustment.
    """
    # the second top-row slot would hold a throughput panel; a latency sweep has
    # no throughput metric, so SAY SO rather than leave dead space (reads as a
    # render bug) or, far worse, fabricate a zero curve.
    cx, cw = _chart_column_pos(1, 2, M, W)
    _chart_panel(el, cx, y, cw, h)  # append rect, discard content box
    _svg_title(el, cx, cw, y, "Throughput", dy=16)
    _svg_placeholder_text(el, cx + cw / 2, y + h / 2 + 5,
                          "N/A for latency test")


def _chart_mid_panel_defs(
    server_cpu_vals: list[float],
    client_cpu_vals: list[float],
    server_mem_vals: list[float],
    client_mem_vals: list[float],
    server_ok: list[bool],
    client_ok: list[bool],
    qf: list[float],
) -> list[tuple[str, str, list[tuple[str, list[float], list[float], str]]]]:
    """Build middle-panel definitions for CPU and memory pressure charts."""
    return [
        (
            "Average CPU Utilization",
            "%",
            [
                _chart_sys_series("server", server_cpu_vals, server_ok, qf, "#dc2626"),
                _chart_sys_series("client", client_cpu_vals, client_ok, qf, "#0891b2"),
            ],
        ),
        (
            "Host Memory Pressure Delta",
            "MiB",
            [
                _chart_sys_series("server", server_mem_vals, server_ok, qf, "#ca8a04"),
                _chart_sys_series("client", client_mem_vals, client_ok, qf, "#9333ea"),
            ],
        ),
    ]


def _init_svg_chart(
    title: str, subtitle: str,
    num_combos: int = 0,
) -> tuple[list[str], int, int, int]:
    """Create an SVG element list and write the opening tag with title/subtitle.

    *num_combos* controls the viewport height so the per-core CPU table at
    the bottom is never clipped (a 12-combo sweep needs more vertical room).
    Returns ``(el, W, H, M)`` — the element accumulator, SVG dimensions,
    and margin constant used for layout throughout the chart.
    """
    W, M = 1180, 16
    # Table starts at y ≈ 794 with th = 30 + 24 * (num_combos + 1).
    # Floor = 1120 (pre-2025 baseline) to keep tiny sweeps compact.
    H = max(1120, 840 + 30 + 24 * (num_combos + 1) + M)
    el: list[str] = []
    el.append(f"<svg xmlns='http://www.w3.org/2000/svg' width='{W}' height='{H}' viewBox='0 0 {W} {H}' style='background:#f8fafc'>")
    el.append(f"<text x='{W/2}' y='26' font-family='system-ui,sans-serif' font-size='18' font-weight='bold' fill='#1e293b' text-anchor='middle'>{html.escape(title)}</text>")
    if subtitle:
        el.append(f"<text x='{W/2}' y='42' font-family='system-ui,sans-serif' font-size='12' fill='#64748b' text-anchor='middle'>{html.escape(subtitle)}</text>")
    return el, W, H, M


def _chart_column_pos(
    col: int,
    num_cols: int,
    margin: float,
    total_width: float,
) -> tuple[float, float]:
    """Compute (left_edge, column_width) for a chart column in a multi-column layout.

    Columns are evenly spaced with *margin* gaps between them:
      left-margin | col 0 | gap | col 1 | ... | right-margin

    This replaces the inline formula used in ``_svg_chart`` so the column-layout
    math is testable in isolation and reusable for any number of columns.
    """
    cw = (total_width - (num_cols + 1) * margin) / num_cols
    cx = margin + col * (cw + margin)
    return cx, cw


def _chart_panel_rect(x: float, y: float, w: float, h: float) -> str:
    """Generate an SVG ``rect`` element for a chart panel with standard styling."""
    return (f"<rect x='{x}' y='{y}' width='{w}' height='{h}' "
            f"rx='6' fill='#fff' stroke='#e2e8f0' stroke-width='1'/>")


def _chart_content_box(px: float, py: float, pw: float, ph: float) -> tuple[float, float, float, float]:
    """Compute the chart content area inset 8/4/8/4 px from a panel rect.

    The chart content area is 8 px narrower on each side and 4 px shorter
    at top and bottom — the same inset used by ``_line_chart``,
    ``_multi_line_chart``, and ``_stacked_bar`` in ``_svg_chart``.
    """
    return (px + 8, py + 4, pw - 16, ph - 8)


def _chart_panel(el: list[str], x: float, y: float, w: float, h: float) -> tuple[float, float, float, float]:
    """Append a panel rect to *el* and return its inner chart content area.

    Combines ``_chart_panel_rect`` (appends the SVG rect element) with
    ``_chart_content_box`` (computes the 8/4/8/4 px inset content box) so
    callers write the panel dimensions only once — eliminating the
    desync hazard of a height literal on adjacent lines.
    """
    el.append(_chart_panel_rect(x, y, w, h))
    return _chart_content_box(x, y, w, h)


def _svg_chart(summary: list[dict[str, Any]], report_config: dict[str, Any] | None = None) -> str:
    """Generate a static SVG report from sweep summary data.

    Caller (``generate_report``) guarantees *summary* is non-empty, so no
    empty-check is duplicated here.
    """
    report_config = report_config or {}
    qps, ok_idx, bw_x, bw_vals, rate_vals, lat_vals, is_latency = _compute_metric_series(summary)

    # run_perftest sets process["server_perf"] BEFORE it reads the client JSON
    # and BEFORE it layers result["error"] on top, so a run whose client perftest
    # failed (or whose metrics failed validation) is on disk carrying BOTH an
    # error AND a populated server-side profile.  Aggregating it into the "Top CPU
    # Consumers" chart would plot a failed run's profile as a valid measurement —
    # the same masquerade ok_idx fixes for the bw/rate curve — so the stacked-bar
    # series (and its bar labels below) is built from successful runs only.
    perf_profiles, _, series = _load_perf_bar_series(summary, ok_idx)

    server_cpu_vals, client_cpu_vals, server_mem_vals, client_mem_vals, qf, server_ok, client_ok, cpu_key = _compute_sys_resource_data(summary, qps)

    # Derive the per-core column set from the UNION of all runs
    table_hdrs, table_rows = _compute_core_table_data(summary, cpu_key, qps, server_ok)

    title, subtitle = _chart_title(summary, report_config)

    el, W, _, M = _init_svg_chart(title, subtitle, num_combos=len(summary))

    y = 56
    # Panel heights — named locals keep the height literal in sync across
    # the _chart_panel() call and the y += cursor advance below.
    TOP_H = 220
    MID_H = 220
    BAR_H = 250

    top_panels = _chart_top_panel_defs(is_latency, lat_vals, bw_vals, rate_vals)
    for col, (label, ylbl, yv, clr, dec) in enumerate(top_panels):
        cx, cw = _chart_column_pos(col, 2, M, W)
        _line_chart(el, bw_x, yv, label, ylbl, *_chart_panel(el, cx, y, cw, TOP_H), clr, dec)
    if is_latency:
        _chart_latency_na_panel(el, M, y, W, TOP_H)

    y += TOP_H + M
    mid_panels = _chart_mid_panel_defs(
        server_cpu_vals, client_cpu_vals, server_mem_vals, client_mem_vals,
        server_ok, client_ok, qf,
    )
    for col, (label, ylbl, chart_series) in enumerate(mid_panels):
        cx, cw = _chart_column_pos(col, 2, M, W)
        _multi_line_chart(el, qf, chart_series, label, ylbl, *_chart_panel(el, cx, y, cw, MID_H))

    y += MID_H + M
    _stacked_bar(el, [str(qps[i]) for i in ok_idx], series, perf_profiles, *_chart_panel(el, M, y, W - 2 * M, BAR_H))

    y += BAR_H + M
    th = 30 + 24 * (len(table_rows) + 1)
    tx, ty, tw, _ = _chart_panel(el, M, y, W - 2 * M, th)
    _svg_table(el, table_hdrs, table_rows, "Server Per-Core CPU Utilization (%)", tx, ty, tw)

    el.append("</svg>")
    return "\n".join(el)


def _scale_y(v: float, cy: float, ch: float, ymn: float, ymx: float) -> float:
    """Convert a data value to a pixel y-coordinate in the chart area."""
    return cy + ch - (v - ymn) / (ymx - ymn) * ch


def _draw_chart_grid(
    el: list[str],
    cx: float, cy: float, cw: float, ch: float,
    ymn: float, ymx: float,
    dec: int,
    x_vals: list[float],
    xmn: float, xmx: float,
) -> None:
    """Draw y-axis grid lines, x-axis ticks, and x labels."""
    _draw_y_grid(el, cx, cy, cw, ch, [_fmt(ymn + (ymx - ymn) * (1 - i / 4), dec) for i in range(5)])
    for vx in x_vals:
        lx = _log2_px(vx, cx, xmn, xmx, cw)
        el.append(f"<line x1='{lx}' y1='{cy + ch}' x2='{lx}' y2='{cy + ch + 4}' stroke='#94a3b8' stroke-width='1'/>")
        _chart_x_label(el, lx, cy, ch, f"{vx:g}")


def _svg_title(el: list[str], x: float, w: float, y: float, title: str,
              dy: float = 12) -> None:
    """Append a bold centered chart title to the SVG element list.

    *dy* is the y-offset from the panel top (default 12).  Pass a custom value
    when a panel needs a different title baseline (e.g. ``dy=16`` for the
    latency N/A placeholder panel).
    """
    el.append(
        f"<text x='{x + w/2}' y='{y + dy}' "
        f"font-family='system-ui,sans-serif' font-size='14' font-weight='bold' "
        f"fill='#1e293b' text-anchor='middle'>{html.escape(title)}</text>"
    )


def _svg_placeholder_text(el: list[str], cx: float, cy: float, msg: str) -> None:
    """Append a centered grey placeholder label at ``(cx, cy)``.

    The muted ``#94a3b8`` note used by panels that have no data to plot
    (latency-test N/A slot, *no valid runs*, *no per-core data*) instead of
    leaving dead space or fabricating a curve.
    """
    el.append(f"<text x='{cx}' y='{cy}' font-family='system-ui,sans-serif' "
              f"font-size='12' fill='#94a3b8' text-anchor='middle'>{html.escape(msg)}</text>")


def _svg_empty_panel(el: list[str], x: float, w: float, y: float,
                     title: str, cy: float, msg: str) -> None:
    """Render a panel with a bold title and a centered grey empty-state note.

    Shared by ``_stacked_bar`` (``y + h / 2``) and ``_svg_table``
    (``y + 28``) — both emit a title followed by a placeholder, then
    ``return`` early so no further panel content is drawn.
    """
    _svg_title(el, x, w, y, title)
    _svg_placeholder_text(el, x + w / 2, cy, msg)


def _chart_legend_item(el: list[str], x: float, y: float, color: str, name: str) -> None:
    """Append a colored legend item (rect + name label) to the SVG element list.

    Draws a small filled ``rect`` (10×10, border-radius 2) at ``(x, y)``
    followed by a 10px ``system-ui`` text label starting 16 px to the right
    and 10 px below the rect — the spacing used by ``_multi_line_chart``
    and ``_stacked_bar``.
    """
    el.append(f"<rect x='{x}' y='{y}' width='10' height='10' fill='{color}' rx='2'/>")
    el.append(f"<text x='{x + 16}' y='{y + 10}' "
              f"font-family='system-ui,sans-serif' font-size='10' fill='#1e293b'>{html.escape(name)}</text>")


def _chart_y_label(el: list[str], x: float, y: float, pl: float, pt: float, ch: float, label: str) -> None:
    """Append a rotated y-axis label to the SVG element list.

    The label is rendered vertically (``rotate(-90)``) centered along the
    chart area's left margin — the same axis-label pattern used by
    ``_draw_chart_header`` and ``_stacked_bar``.
    """
    el.append(f"<text x='{x + pl/2}' y='{y + pt + ch/2}' "
              f"font-family='system-ui,sans-serif' font-size='11' "
              f"fill='#64748b' text-anchor='middle' "
              f"transform='rotate(-90,{x + pl/2},{y + pt + ch/2})'>{html.escape(label)}</text>")


def _chart_x_label(el: list[str], cx: float, cy: float, ch: float, text: str) -> None:
    """Append a centered x-axis tick label 18 px below the plot baseline.

    The muted 10px ``#64748b`` label shared by ``_draw_chart_grid`` (numeric
    log2-axis ticks) and ``_stacked_bar`` (per-bar category labels). *text*
    must already be the final display string; callers HTML-escape as needed.
    """
    el.append(f"<text x='{cx}' y='{cy + ch + 18}' "
              f"font-family='system-ui,sans-serif' font-size='10' "
              f"fill='#64748b' text-anchor='middle'>{text}</text>")


def _draw_chart_header(
    el: list[str],
    title: str, ylb: str,
    x: float, y: float, w: float,
    pl: float, pt: float, ch: float,
) -> None:
    """Append the chart title (top-centered) and y-axis label (rotated)."""
    _svg_title(el, x, w, y, title)
    _chart_y_label(el, x, y, pl, pt, ch, ylb)


def _log2_px(v: float, cx: float, xmn: float, xmx: float, cw: float) -> float:
    """Log2-scaled x-position for SVG chart points."""
    if xmx != xmn:
        return cx + (math.log2(v) - math.log2(xmn)) / (math.log2(xmx) - math.log2(xmn)) * cw
    return cx + cw / 2


def _filter_positive_x(xv: list[float], yv: list[float]) -> list[tuple[float, float]]:
    """Filter (x, y) data-point pairs to those with *x* > 0.

    The SVG chart functions use ``_log2_px`` which calls ``math.log2(vx)``,
    raising ``ValueError`` on zeros and negatives.  This guard drops such
    points so the chart renders correctly; the per-run data in the CSV and
    summary JSON remains the authoritative record regardless.
    """
    return [(vx, vy) for vx, vy in zip(xv, yv) if vx > 0]


def _chart_empty_state(
    el: list[str],
    title: str,
    ylb: str,
    x: float,
    y: float,
    w: float,
    h: float,
    pl: float,
    pt: float,
    ch: float,
) -> None:
    """Draw a chart frame with a 'no valid runs' empty-state message.

    Renders the chart header and a centred label in the chart body rather
    than crashing on ``ZeroDivisionError`` or fabricating a plausible-looking
    line from no data.  Used by ``_line_chart`` / ``_multi_line_chart`` when
    every point on an axis was dropped (e.g. non-positive qp values that the
    log2 axis cannot represent).
    """
    _draw_chart_header(el, title, ylb, x, y, w, pl, pt, ch)
    _svg_placeholder_text(el, x + w / 2, y + h / 2, "no valid runs")


def _chart_area(
    x: float, y: float, w: float, h: float,
    pl: float, pr: float, pb: float, pt: float,
) -> tuple[float, float, float, float]:
    """Compute the chart drawing area from margins.

    Returns ``(cx, cy, cw, ch)`` — the top-left corner and dimensions of the
    inner plotting area inside the chart frame defined by margins *pl*, *pr*,
    *pb*, *pt* (left, right, bottom, top).
    """
    cx = x + pl
    cy = y + pt
    cw = w - pl - pr
    ch = h - pt - pb
    return cx, cy, cw, ch


def _draw_series_polyline(
    el: list[str],
    pts: list[tuple[float, float]],
    cx: float, cy: float, cw: float, ch: float,
    xmn: float, xmx: float, ymn: float, ymx: float,
    clr: str,
    r: float = 4,
) -> None:
    """Draw a data series as a log2-scaled polyline plus circle markers.

    *pts* is the already-positive-filtered list of (x, y) data pairs.
    Shared by ``_line_chart`` (per-point value labels, r=4) and
    ``_multi_line_chart`` (no labels, r=3.5).
    """
    poly = " ".join(f"{_log2_px(vx, cx, xmn, xmx, cw)},{_scale_y(vy, cy, ch, ymn, ymx)}" for vx, vy in pts)
    el.append(f"<polyline points='{poly}' fill='none' stroke='{clr}' stroke-width='2.5' stroke-linejoin='round'/>")
    for vx, vy in pts:
        dx, dy = _log2_px(vx, cx, xmn, xmx, cw), _scale_y(vy, cy, ch, ymn, ymx)
        el.append(f"<circle cx='{dx}' cy='{dy}' r='{r}' fill='{clr}'/>")


def _line_chart(el: list[str], xv: list[float], yv: list[float], title: str, ylb: str, x: float, y: float, w: float, h: float, clr: str, dec: int = 0) -> None:
    pl, pr, pb, pt = 50, 20, 40, 40
    cx, cy, cw, ch = _chart_area(x, y, w, h, pl, pr, pb, pt)
    pts = _filter_positive_x(xv, yv)
    if not pts:
        # Every run on this axis errored — draw empty state rather than fabricating data.
        _chart_empty_state(el, title, ylb, x, y, w, h, pl, pt, ch)
        return
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    ymn = 0
    ymx = max(ys) * 1.1
    if ymx <= 0:
        ymx = 1
    xmn, xmx = min(xs), max(xs)

    _draw_chart_header(el, title, ylb, x, y, w, pl, pt, ch)
    _draw_chart_grid(el, cx, cy, cw, ch, ymn, ymx, dec, xs, xmn, xmx)
    _draw_series_polyline(el, pts, cx, cy, cw, ch, xmn, xmx, ymn, ymx, clr, 4)
    for vx, vy in pts:
        dx, dy = _log2_px(vx, cx, xmn, xmx, cw), _scale_y(vy, cy, ch, ymn, ymx)
        el.append(f"<text x='{dx}' y='{dy - 10}' font-family='system-ui,sans-serif' font-size='10' fill='#1e293b' text-anchor='middle'>{_fmt(vy, dec)}</text>")


def _multi_line_chart(
    el: list[str],
    xv: list[float],
    series: list[tuple[str, list[float], list[float], str]],
    title: str,
    ylb: str,
    x: float,
    y: float,
    w: float,
    h: float,
) -> None:
    # Each series carries its OWN x-points (name, xs, ys, colour): a series may
    # omit points (e.g. a failed sys grab) without fabricating a 0 at that qp.
    # ``xv`` is the shared axis spanning every sweep x-value; non-positive values
    # are dropped (the log2 axis can't represent them) before it drives ticks/scale.
    pl, pr, pb, pt = 50, 90, 40, 40
    cx, cy, cw, ch = _chart_area(x, y, w, h, pl, pr, pb, pt)
    # Drop non-positive shared-axis values (e.g. qp=0 from _expand's default
    # from=0) so the axis tick loop renders instead of crashing on math.log2(0).
    axis = [v for v in xv if v > 0]
    values = [v for _, _xs, vals, _ in series for v in vals]
    ymn = min(0, min(values)) if values else 0
    ymx = max(values) * 1.15 if values else 1
    if ymx <= 0:
        ymx = 1
    if not axis:
        # No plottable shared-axis values — draw empty state rather than crashing.
        _chart_empty_state(el, title, ylb, x, y, w, h, pl, pt, ch)
        return
    xmn, xmx = min(axis), max(axis)

    _draw_chart_header(el, title, ylb, x, y, w, pl, pt, ch)
    _draw_chart_grid(el, cx, cy, cw, ch, ymn, ymx, 1, axis, xmn, xmx)
    for si, (name, xs, vals, clr) in enumerate(series):
        spts = _filter_positive_x(xs, vals)
        _draw_series_polyline(el, spts, cx, cy, cw, ch, xmn, xmx, ymn, ymx, clr, 3.5)
        ly = cy + 4 + si * 18
        _chart_legend_item(el, cx + cw + 14, ly, clr, name)


def _stacked_bar(el: list[str], xlb: list[str], series: list[tuple[str, list[float]]], perf_profiles: list[dict[str, Any] | None], x: float, y: float, w: float, h: float) -> None:
    pl, pr, pb, pt = 50, 160, 40, 40
    cx, cy, cw, ch = _chart_area(x, y, w, h, pl, pr, pb, pt)
    n = len(xlb)
    if n == 0:
        # every run errored → no valid profile to chart; an empty-state note
        # avoids both a ZeroDivisionError (cw / n) and a misleadingly blank panel.
        _svg_empty_panel(el, x, w, y, "Top CPU Consumers (self %)", y + h / 2, "no valid runs")
        return
    bw = min(cw / n * 0.7, 50)
    gap = (cw - bw * n) / (n + 1)
    _chart_y_label(el, x, y, pl, pt, ch, "Self %")
    _draw_y_grid(el, cx, cy, cw, ch, [str(100 - 100 * i // 4) for i in range(5)])
    for si in range(n):
        bx = cx + gap + si * (bw + gap)
        _chart_x_label(el, bx + bw / 2, cy, ch, html.escape(xlb[si]))
        # Mark columns whose profile is unavailable — corrupt/missing result.json,
        # or a profile that was attempted but failed to collect (perf start or
        # stop/report error) — so they stay visually distinct from a run that
        # genuinely had no top consumers (a real but empty profile).
        if si < len(perf_profiles) and perf_profiles[si] is None:
            el.append(
                f"<rect x='{bx}' y='{cy}' width='{bw}' height='{ch}' "
                f"fill='#e2e8f0' stroke='#94a3b8' stroke-width='1' stroke-dasharray='4,2'/>"
            )
            el.append(
                f"<text x='{bx + bw / 2}' y='{cy + ch / 2}' "
                f"font-family='system-ui,sans-serif' font-size='9' fill='#64748b' "
                f"text-anchor='middle'>n/a</text>"
            )
            continue
        bottom = 0.0
        for ci, (_, vals) in enumerate(series):
            v = vals[si]
            # y-axis is fixed 0-100 % (self-% values are bounded); clamp each
            # segment so the total stack never overflows the chart viewport.
            remaining = max(0, 100.0 - bottom)
            v_clip = max(0.0, min(v, remaining))
            if v_clip > 0:
                bh = v_clip / 100 * ch
                clr = SVG_COLORS[ci % len(SVG_COLORS)]
                el.append(f"<rect x='{bx}' y='{cy + ch - bottom - bh}' width='{bw}' height='{bh}' fill='{clr}'/>")
            bottom += v_clip
    lx, ly2 = cx + cw + 12, cy + 4
    for ci, (name, _) in enumerate(series):
        display = name if len(name) < 32 else name[:29] + "..."
        _chart_legend_item(el, lx, ly2, SVG_COLORS[ci % len(SVG_COLORS)], display)
        ly2 += 18


def _svg_cell_text(el: list[str], x: float, cw: float, ci: int, y: float,
                   text: str, bold: bool = True) -> None:
    """Append a table cell text element with centered ``font-size='11'`` text.

    Shared by the column-header row and all data rows in ``_svg_table`` to
    keep the ``font-family`` / ``font-size`` / ``fill`` style block in one
    place rather than duplicated on adjacent lines.
    """
    el.append(
        f"<text x='{x + ci * cw + cw / 2}' y='{y + 16}' "
        f"font-family='system-ui,sans-serif' font-size='11' "
        f"font-weight='{'bold' if bold else 'normal'}' "
        f"fill='#1e293b' text-anchor='middle'>{html.escape(text)}</text>"
    )


def _svg_cell_rect(el: list[str], x: float, cw: float, ci: int, y: float,
                   fill: str) -> None:
    """Append a table cell background rect.

    Companion to ``_svg_cell_text`` — both share the same column-geometry
    parameters so the cell background and its label stay in sync.  Used by
    the column-header row (always ``#f1f5f9``) and alternating data rows
    (``#f8fafc`` on odd indices) in ``_svg_table``.
    """
    el.append(f"<rect x='{x + ci * cw}' y='{y}' width='{cw}' height='24' fill='{fill}'/>")


def _svg_table(el: list[str], hdrs: list[str], rows: list[list[str]], title: str, x: float, y: float, w: float) -> None:
    if not hdrs:
        _svg_empty_panel(el, x, w, y, title, y + 28, "no per-core data")
        return
    ncols = len(hdrs)
    cw = w / ncols
    _svg_title(el, x, w, y, title)
    ty = y + 28
    for ci, hdr in enumerate(hdrs):
        _svg_cell_rect(el, x, cw, ci, ty, "#f1f5f9")
        _svg_cell_text(el, x, cw, ci, ty, hdr, bold=True)
        el.append(f"<line x1='{x + ci * cw}' y1='{ty}' x2='{x + ci * cw}' y2='{ty + 24 * (len(rows) + 1)}' stroke='#e2e8f0' stroke-width='0.5'/>")
    el.append(f"<line x1='{x + ncols * cw}' y1='{ty}' x2='{x + ncols * cw}' y2='{ty + 24 * (len(rows) + 1)}' stroke='#e2e8f0' stroke-width='0.5'/>")
    for ri, row in enumerate(rows):
        ry = ty + 24 * (ri + 1)
        bg = "#f8fafc" if ri % 2 == 1 else ""
        for ci, val in enumerate(row):
            if bg:
                _svg_cell_rect(el, x, cw, ci, ry, bg)
            _svg_cell_text(el, x, cw, ci, ry, val, bold=(ci == 0))


def _fmt(v: float, d: int = 0) -> str:
    return f"{v:.0f}" if v >= 1000 else f"{v:.{d}f}"


def _generate_pdf(svg_path: Path, out_dir: Path) -> None:
    """Convert SVG report to PDF using ``cairosvg`` if available.

    Silently skips PDF generation when ``cairosvg`` is not installed.
    """
    try:
        cairosvg_path = shutil.which("cairosvg")
        if cairosvg_path is not None:
            pdf_path = out_dir / "chart.pdf"
            subprocess.run([cairosvg_path, str(svg_path), "-o", str(pdf_path)], check=True)
            print(f"Report PDF → {pdf_path}")
    except Exception as exc:
        print(f"PDF generation skipped: {exc}", file=sys.stderr)


def generate_report(output_dir: str) -> None:
    """Generate SVG (and optionally PDF) report from existing sweep results."""
    out = Path(output_dir)
    summary = _read_json(out / "summary.json")
    if not summary:
        raise ValueError("summary.json contains no sweep entries")
    for i, entry in enumerate(summary):
        entry["_result_path"] = str(out / f"{i+1:04d}" / "result.json")
    report_config: dict[str, Any] = {}
    run_config_path = out / "run_config.json"
    if run_config_path.exists():
        report_config = _get_dict(_read_json(run_config_path), "report")
    svg = _svg_chart(summary, report_config=report_config)
    svg_path = out / "chart.svg"
    svg_path.write_text(svg)
    print(f"Report SVG → {svg_path}")
    _generate_pdf(svg_path, out)


def _dry_run_print(config: dict[str, Any]) -> None:
    """Print every parameter combo that would run, without executing."""
    count = 0
    for combo in sweep_config(config):
        args = _build_args(combo)
        desc = " ".join(f"{k}={v}" for k, v in sorted(combo.items()))
        print(f"  {desc:50s}  {' '.join(args)}")
        count += 1
    print(f"\nTotal: {count} combo(s)", flush=True)


def _count_summary_errors(output_dir: str | Path) -> int:
    """Count runs with errors in the sweep summary.

    Reads ``summary.json`` from *output_dir* and returns the number of
    entries whose ``error`` field is non-None.  Returns 0 when the file
    is missing or corrupt so callers always get a usable count.
    """
    summary_path = Path(output_dir) / "summary.json"
    try:
        summary: list[dict[str, Any]] = _read_json(summary_path)
    except (FileNotFoundError, json.JSONDecodeError):
        return 0
    if not isinstance(summary, list):
        return 0
    return sum(1 for entry in summary if entry.get("error"))


def _exit_error(msg: str) -> NoReturn:
    """Print *msg* to stderr and exit with status 1."""
    print(msg, file=sys.stderr)
    sys.exit(1)


def _load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load and parse a YAML sweep config file.

    Returns the parsed config dict.
    Exits the program with an error message on failure.
    """
    try:
        config = yaml.safe_load(Path(path).read_text())
    except FileNotFoundError:
        _exit_error(f"ERROR: config file not found: {path}")
    except PermissionError:
        _exit_error(f"ERROR: permission denied reading config file: {path}")
    except yaml.YAMLError as exc:
        _exit_error(f"ERROR: malformed YAML in {path}: {exc}")
    except OSError as exc:
        _exit_error(f"ERROR: cannot read config file {path}: {exc}")
    return cast(dict[str, Any], config)


def _validate_sweep_config(config: dict[str, Any]) -> None:
    """Validate the sweep config has the expected structure.

    Exits the program with an error message on validation failure.
    """
    if not isinstance(config, dict):
        _exit_error("ERROR: config file is empty or not a YAML mapping")

    sweep = config.get("sweep")
    if not isinstance(sweep, list):
        _exit_error(
            f"ERROR: config 'sweep' must be a list, got {type(sweep).__name__!r}",
        )
    if not sweep:
        _exit_error("ERROR: config 'sweep' list is empty")


def _build_cli_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for the sweep tool."""
    ap = argparse.ArgumentParser(
        description="RDMA perftest parameter sweep tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--config", "-c", help="YAML sweep config (run sweep mode)")
    ap.add_argument("--output-dir", "-o", default="sweep_results", help="Output directory")
    ap.add_argument("--report", "-r", nargs="?", const=True, default=False, help="Generate report from existing results (optional: path to results dir)")
    ap.add_argument("--dry-run", "-n", action="store_true", help="Print all parameter combos and exit (no execution)")
    return ap


def main() -> None:
    ap = _build_cli_parser()
    args = ap.parse_args()

    report_dir = args.report if isinstance(args.report, str) else args.output_dir
    if args.report:
        generate_report(report_dir)
        return

    if yaml is None:
        _exit_error("ERROR: PyYAML is required.  pip install pyyaml")

    if not args.config:
        ap.print_help(file=sys.stderr)
        sys.exit(1)

    config = _load_yaml_config(args.config)
    _validate_sweep_config(config)

    if args.dry_run:
        _dry_run_print(config)
        return

    # Pre-populate local host set for self-detection before first SSH use.
    init_local_hosts()
    run_sweep(config, output_dir=args.output_dir)

    # Exit non-zero when any individual run produced an error.
    errors = _count_summary_errors(args.output_dir)
    if errors:
        _exit_error(
            f"ERROR: {errors} run(s) failed — check summary.csv for details",
        )


if __name__ == "__main__":
    main()
