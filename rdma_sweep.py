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
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]

from rdma_config import (
    DEFAULT_PERFTEST_CONFIG,
    parse_bool,
    resolve_perftest_paths,
    runtime_config as _runtime_config,
)
from rdma_remote import (
    init_local_hosts,
    run_remote as _run_remote,
    run_remote_result as _run_remote_result,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def timestamp() -> str:
    """ISO-8601 UTC timestamp (nanosecond-precision, Python 3.12 compat)."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_size(s: str) -> int:
    """Parse human-readable sizes like '1M', '64K', '2G' → bytes."""
    s = s.strip().upper()
    multipliers = {"K": 1024, "M": 1024**2, "G": 1024**3, "B": 1}
    for suffix, mul in multipliers.items():
        if s.endswith(suffix):
            return int(float(s[:-1]) * mul)
    return int(s)


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
        return _run_remote(
            cmd,
            self.host,
            timeout=10,
            ssh_config=self.ssh_config,
            sudo=False,
        ).splitlines()

    def grab(self) -> dict[str, Any]:
        """Return a snapshot of /proc/stat, /proc/softirqs, /proc/meminfo."""
        try:
            stat_lines = self._run(f"cat {self.PROC_STAT}")
            mem_lines = self._run(f"cat {self.PROC_MEMINFO}")
            sirq_lines = self._run(f"cat {self.PROC_SOFTIRQS}")
        except Exception as exc:
            return {"error": str(exc)}

        cores: dict[str, dict[str, int]] = {}
        for line in stat_lines:
            parts = line.split()
            if parts[0].startswith("cpu"):
                vals = [int(v) for v in parts[1:]]
                cores[parts[0]] = {
                    k: v for k, v in
                    zip(["user", "nice", "system", "idle", "iowait",
                         "irq", "softirq", "steal"], vals)
                }

        softirqs: dict[str, int] = {}
        for line in sirq_lines:
            parts = line.split()
            if parts[0].startswith("CPU"):
                # header line "CPU0 CPU1 CPU2 ..." — skip
                continue
            # line like "NET_RX:  12345  0  67890  0  ..."
            name = parts[0].rstrip(":")
            vals = [int(v) for v in parts[1:]]
            softirqs[name] = sum(vals)

        mem: dict[str, int] = {}
        for line in mem_lines:
            if ":" in line:
                key = line.split(":")[0].strip()
                val = line.split(":")[1].strip().split()[0]
                try:
                    mem[key] = int(val)
                except ValueError:
                    pass

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
        be = before.get("softirqs", {})
        ae = after.get("softirqs", {})
        return {k: ae[k] - be.get(k, 0) for k in ae if k in be}

    @staticmethod
    def compute_cpu_diff(
        before: dict[str, Any],
        after: dict[str, Any],
    ) -> dict[str, float]:
        """Return per-core utilisation (%) between two grabs."""
        result: dict[str, float] = {}
        be = before.get("cores", {})
        ae = after.get("cores", {})
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
                result[key] = 100.0 * (1 - (a["idle"] - b["idle"]) / dt)
        return result

    @staticmethod
    def extract_mem(after: dict[str, Any]) -> dict[str, str]:
        """Return key memory metrics from a grab (in human-readable form)."""
        m = after.get("mem_kB", {})
        if not m:
            return {}
        return {
            "MemTotal":  format_size(m.get("MemTotal", 0) * 1024),
            "MemFree":   format_size(m.get("MemFree", 0) * 1024),
            "MemUsed":   format_size((m.get("MemTotal", 0) - m.get("MemFree", 0) - m.get("Buffers", 0) - m.get("Cached", 0)) * 1024),
            "Buffers":   format_size(m.get("Buffers", 0) * 1024),
            "Cached":    format_size(m.get("Cached", 0) * 1024),
        }

    @staticmethod
    def compute_mem_delta(
        before: dict[str, Any],
        after: dict[str, Any],
    ) -> dict[str, str]:
        """Return delta memory (after - before) in human-readable form."""
        b = before.get("mem_kB", {})
        a = after.get("mem_kB", {})
        if not a or not b:
            return {}
        def _used(m: dict[str, int]) -> int:
            return m.get("MemTotal", 0) - m.get("MemFree", 0) - m.get("Buffers", 0) - m.get("Cached", 0)
        used_delta_kb = _used(a) - _used(b)
        return {
            "MemUsedDelta": format_size(used_delta_kb * 1024),
            "MemFreeDelta": format_size((a.get("MemFree", 0) - b.get("MemFree", 0)) * 1024),
        }


# ---------------------------------------------------------------------------
# Perftest runner
# ---------------------------------------------------------------------------

PERFTEST_BINS = [
    "ib_write_bw", "ib_read_bw", "ib_send_bw",
    "ib_write_lat", "ib_read_lat", "ib_send_lat",
]

def _wait_for_port(
    host: str,
    port: int,
    timeout: int = 30,
    ssh_config: dict[str, Any] | None = None,
) -> None:
    """Poll *host* until *port* is listening (or *timeout* seconds elapse)."""
    for _ in range(timeout * 2):
        out = _run_remote(
            f"ss -H -tln 'sport = :{int(port)}' 2>/dev/null | grep -q . && echo ready",
            host,
            check=False,
            ssh_config=ssh_config,
            sudo=False,
        )
        if "ready" in out:
            return
        time.sleep(0.5)
    raise TimeoutError(f"{host}:{port} did not listen within {timeout}s")


def _env_prefix(perftest_config: dict[str, Any]) -> str:
    env = {
        str(k): str(v)
        for k, v in (perftest_config.get("env", {}) or {}).items()
        if v is not None
    }
    rdma_core_lib = str(perftest_config.get("rdma_core_lib", "") or "")
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


def _has_flag(args: list[str], *flags: str) -> bool:
    flag_set = set(flags)
    for arg in args:
        if arg in flag_set:
            return True
        for flag in flag_set:
            if arg.startswith(f"{flag}="):
                return True
    return False


def _port_from_args(extra_args: list[str], default_port: int) -> int:
    for i in range(0, len(extra_args) - 1):
        if extra_args[i] == "-p":
            return int(extra_args[i + 1])
    for arg in extra_args:
        if arg.startswith("--port="):
            return int(arg.split("=", 1)[1])
    return default_port


def _parse_perf_report(raw: str) -> dict[str, float]:
    """Parse ``perf report --stdio --no-header -g none`` into {symbol: self_pct}.

    Only includes symbols with non-zero self overhead.  The self column is the
    second percentage field (first is children).  Symbols from different DSOs
    sharing the same name are disambiguated as "symbol@dso".
    """
    result: dict[str, float] = {}
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Top-level line: "  children%  self%  cmd  shared_obj  [annotation] symbol"
        # Must contain [k] or [.] annotation
        if "[k]" not in line and "[.]" not in line:
            continue
        parts = line.split()
        # Expected: children%, self%, cmd, shared_obj, [annotation], symbol...
        if len(parts) < 6:
            continue
        self_pct_str = parts[1].rstrip("%")
        try:
            self_pct = float(self_pct_str)
        except ValueError:
            continue
        if self_pct == 0.0:
            continue
        # Symbol may contain spaces (demangled C++ names).  Extract text
        # after the [k] or [.] annotation marker.
        #   parts[0..4] = children%, self%, cmd, shared_obj, [annotation]
        #   parts[5..]  = symbol tokens
        marker_idx = next(i for i, p in enumerate(parts) if p in ("[k]", "[.]"))
        dso = parts[3]
        symbol = " ".join(parts[marker_idx + 1:])
        # Some symbols (e.g., poll) can appear from different DSOs.
        # Always disambiguate with @dso so the key is unique.
        key = f"{symbol}@{dso}"
        # Shorten kernel DSO to keep keys readable.
        if dso == "[kernel.kallsyms]":
            key = symbol
        result[key] = self_pct
    return result


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
    wait_timeout = int(perftest_config.get("wait_timeout", 30))
    default_port = int(perftest_config.get("default_port", 18515))
    env = _env_prefix(perftest_config)
    env_cmd = f"{env} " if env else ""

    # Build args: strip out any --out_json (we add our own).
    filtered = _filtered_perftest_args(extra_args)
    args_str = " ".join(shlex.quote(a) for a in filtered)
    duration_arg = "" if _has_flag(filtered, "-n", "--iters") else f" -D {int(duration)}"
    bin_q = shlex.quote(bin_path)
    server_pid_q = shlex.quote(server_pid_file)
    server_log_q = shlex.quote(server_log_file)
    json_q = shlex.quote(json_file)
    time_q = shlex.quote(time_file)
    perf_data_q = shlex.quote(perf_data)
    perf_pid_q = shlex.quote(perf_pid_file)
    tmp_dir_q = shlex.quote(str(perftest_config["tmp_dir"]))

    # perftest starts server mode by omitting the remote address.
    server_cmd = f"{env_cmd}{bin_q} {args_str}{duration_arg}".strip()
    process: dict[str, Any] = {
        "run_id": run_id,
        "server_host": server_host,
        "client_host": client_host,
        "server_address": server_address,
        "server_pid": None,
        "server_perf": {},
        "client_usage": {},
        "commands": {},
    }

    def _server_log_tail() -> str:
        log = _run_remote_result(
            f"tail -200 {server_log_q} 2>/dev/null || true",
            server_host,
            ssh_config=ssh_config,
            sudo=False,
        )
        process["commands"]["server_log_tail"] = log.to_dict()
        return log.stdout.strip()

    try:
        server_start = _run_remote_result(
            f"mkdir -p {tmp_dir_q}; rm -f {server_pid_q} {server_log_q}; "
            f"{server_cmd} >{server_log_q} 2>&1 & "
            f"pid=$!; "
            f"started=$(ps -p $pid -o lstart= 2>/dev/null | sed 's/^ *//'); "
            f"printf '%s\\n%s\\n%s\\n' \"$pid\" \"$started\" {shlex.quote(run_id)} > {server_pid_q}",
            server_host,
            ssh_config=ssh_config,
        )
        process["commands"]["server_start"] = server_start.to_dict()
        if not server_start.ok:
            return {"error": f"server launch failed: {server_start.error_summary()}", "_process": process}
        port = _port_from_args(extra_args, default_port)
        _wait_for_port(server_host, port, timeout=wait_timeout, ssh_config=ssh_config)

        # Capture server PID
        server_pid_read = _run_remote_result(
            f"cat {server_pid_q}",
            server_host,
            ssh_config=ssh_config,
            sudo=False,
        )
        process["commands"]["server_pid_read"] = server_pid_read.to_dict()
        server_pid_raw = server_pid_read.stdout.splitlines()[0].strip() if server_pid_read.stdout.strip() else ""
        server_pid = int(server_pid_raw) if server_pid_raw and server_pid_raw.isdecimal() else None
        process["server_pid"] = server_pid

        # Start perf record -g on server PID (callchain sampling)
        perf_started = False
        if server_pid and perf_record:
            perf_start = _run_remote_result(
                f"rm -f {perf_data_q} {perf_pid_q}; "
                f"perf record -g -p {server_pid} -F 99 -o {perf_data_q} "
                f">/dev/null 2>&1 & echo $! > {perf_pid_q}",
                server_host,
                ssh_config=ssh_config,
            )
            process["commands"]["perf_start"] = perf_start.to_dict()
            perf_started = perf_start.ok
            if not perf_started:
                return {"error": f"perf record failed: {perf_start.error_summary()}", "_process": process}

        # Run client wrapped with /usr/bin/time for process resource accounting.
        time_fmt = "%U %S %P %M %c %w"
        client_run = _run_remote_result(
            f"mkdir -p {tmp_dir_q}; rm -f {json_q} {time_q}; "
            f"{env_cmd}/usr/bin/time --format={shlex.quote(time_fmt)} "
            f"{bin_q} {shlex.quote(server_address)} {args_str}{duration_arg} "
            f"--out_json --out_json_file={json_q} 2>{time_q}",
            client_host,
            timeout=int(duration) + 60,
            ssh_config=ssh_config,
        )
        process["commands"]["client_run"] = client_run.to_dict()

        # Parse /usr/bin/time output (client-side user/sys, RSS)
        proc_usage: dict[str, Any] = {}
        time_read = _run_remote_result(
            f"cat {time_q} 2>/dev/null || true",
            client_host,
            ssh_config=ssh_config,
            sudo=False,
        )
        process["commands"]["client_time_read"] = time_read.to_dict()
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

        # Stop perf record (SIGINT flushes data and exits gracefully)
        perf_report: dict[str, float] = {}
        if perf_started:
            perf_stop = _run_remote_result(
                f"ppid=$(cat {perf_pid_q} 2>/dev/null) && "
                f"kill -INT $ppid 2>/dev/null; "
                f"sleep 2; "  # wait for perf to finalise perf.data
                f"rm -f {perf_pid_q}",
                server_host,
                timeout=15,
                ssh_config=ssh_config,
            )
            process["commands"]["perf_stop"] = perf_stop.to_dict()
            if not perf_stop.ok:
                return {"error": f"perf stop failed: {perf_stop.error_summary()}", "_process": process}
            perf_report_read = _run_remote_result(
                f"perf report --stdio --no-header -g none -i {perf_data_q}",
                server_host,
                ssh_config=ssh_config,
            )
            process["commands"]["perf_report"] = perf_report_read.to_dict()
            if not perf_report_read.ok:
                return {"error": f"perf report failed: {perf_report_read.error_summary()}", "_process": process}
            _run_remote_result(
                f"rm -f {perf_data_q}",
                server_host,
                ssh_config=ssh_config,
            )
            perf_report = _parse_perf_report(perf_report_read.stdout)
        process["server_perf"] = perf_report

        # Read JSON result
        json_read = _run_remote_result(
            f"cat {json_q} 2>/dev/null || true",
            client_host,
            ssh_config=ssh_config,
            sudo=False,
        )
        process["commands"]["client_json_read"] = json_read.to_dict()
        result = _parse_json_output(json_read.stdout)
        if not client_run.ok:
            result["error"] = f"client run failed: {client_run.error_summary()}"
        elif not json_read.ok:
            result["error"] = f"client JSON read failed: {json_read.error_summary()}"
        elif metric_error := _validate_perftest_metrics(binary, result):
            result["error"] = metric_error
        result["_process"] = process
        return result
    except Exception as exc:
        process["exception"] = str(exc)
        try:
            tail = _server_log_tail()
            if tail:
                process["server_log_tail"] = tail
        except Exception:
            pass
        return {"error": str(exc), "_process": process}
    finally:
        cleanup = _cancel(server_host, ssh_config, server_pid_file, bin_path, run_id)
        if cleanup:
            process["commands"]["cleanup"] = cleanup
            cleanup_error = cleanup.get("error")
            if cleanup_error:
                process["cleanup_error"] = cleanup_error


def _cancel(
    host: str,
    ssh_config: dict[str, Any] | None = None,
    server_pid_file: str = "",
    expected_binary: str = "",
    expected_run_id: str = "",
) -> dict[str, Any]:
    evidence: dict[str, Any] = {}
    if not server_pid_file or not expected_binary or not expected_run_id:
        return evidence
    pid_file_q = shlex.quote(server_pid_file)
    try:
        pid_read = _run_remote_result(
            f"cat {pid_file_q} 2>/dev/null || true",
            host,
            ssh_config=ssh_config,
            sudo=False,
        )
        evidence["pid_read"] = pid_read.to_dict()
        pid_lines = pid_read.stdout.splitlines()
        pid = pid_lines[0].strip() if len(pid_lines) >= 1 else ""
        expected_start = pid_lines[1].strip() if len(pid_lines) >= 2 else ""
        pid_run_id = pid_lines[2].strip() if len(pid_lines) >= 3 else ""
        if pid:
            pid_q = shlex.quote(pid)
            expected_q = shlex.quote(expected_binary)
            expected_start_q = shlex.quote(expected_start)
            expected_run_id_q = shlex.quote(expected_run_id)
            pid_run_id_q = shlex.quote(pid_run_id)
            cleanup = _run_remote_result(
                f"args=$(ps -p {pid_q} -o args= 2>/dev/null || true); "
                f"started=$(ps -p {pid_q} -o lstart= 2>/dev/null | sed 's/^ *//'); "
                f"if [ {pid_run_id_q} = {expected_run_id_q} ] && "
                f"[ -n {expected_start_q} ] && [ \"$started\" = {expected_start_q} ] && "
                f"printf '%s\\n' \"$args\" | grep -F -- {expected_q} >/dev/null; then "
                f"kill -TERM {pid_q} 2>/dev/null || true; "
                f"sleep 1; "
                f"kill -KILL {pid_q} 2>/dev/null || true; "
                f"if ps -p {pid_q} >/dev/null 2>&1; then "
                f"echo \"cleanup failed for pid {pid_q}\" >&2; exit 1; fi; "
                f"rm -f {pid_file_q}; echo cleaned; "
                f"else echo \"skip cleanup for pid {pid_q}: $args\" >&2; fi",
                host,
                ssh_config=ssh_config,
            )
            evidence["cleanup"] = cleanup.to_dict()
            if not cleanup.ok:
                evidence["error"] = cleanup.error_summary()
    except Exception as exc:
        evidence["error"] = str(exc)
    return evidence


def _parse_json_output(raw: str | None) -> dict[str, Any]:
    if not raw or raw.strip() == "":
        return {"error": "no JSON output"}

    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
        return {"result": obj}
    except json.JSONDecodeError:
        return {"error": f"JSON parse failed", "raw_snippet": raw[:500]}


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
        return list(param["values"])

    lo = param.get("from", 0)
    hi = param["to"]
    step = param.get("step", 1)
    # keep inclusive
    r: list[int] = []
    v = lo
    while v <= hi:
        r.append(v)
        v += step
    return r


def sweep_config(config: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Yield every parameter combination from the config.

    The config has the structure::

        sweep:
          test: ib_write_bw
          duration: 10
          server:
            host: dpu-server
            address: rdma-server-data.example.com
          client:
            host: dpu-client
          perftest:
            dir: /path
          use_gpu: false
          fixed:
            port: 18515
          sweep:
            - name: msg_size
              values: [1, 4K, 64K, 1M]
            - name: qp
              from: 1
              to: 512
              step: 2x       # ← geometric progression (we'll handle outside)
    """
    sweep_params = config.get("sweep", [])
    fixed = config.get("fixed", {})

    expanded: list[tuple[str, list[Any]]] = []
    for sp in sweep_params:
        name = sp["name"]
        expanded.append((name, _expand(sp)))

    keys = [e[0] for e in expanded]
    for values in itertools.product(*[e[1] for e in expanded]):
        combo: dict[str, Any] = {}
        combo.update(fixed)
        for k, v in zip(keys, values):
            combo[k] = v
        yield combo


def _build_args(combo: dict[str, Any]) -> list[str]:
    """Convert a parameter combo into perftest CLI arguments."""
    args: list[str] = []
    flag_map = {
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
    skip = {
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
    }
    for k, v in combo.items():
        if k in skip:
            continue
        flag = flag_map.get(k, f"--{k.replace('_', '-')}")
        if isinstance(v, bool):
            if v:
                args.append(flag)
        else:
            args.append(flag)
            args.append(str(v))
    return args


def run_sweep(config: dict[str, Any], output_dir: str = "sweep_results") -> list[Path]:
    """Execute the full sweep defined by *config*.

    Returns paths to per-combination JSON result files.
    """
    runtime = _runtime_config(config)
    test = runtime["test"]
    server_host = runtime["server"]["host"]
    client_host = runtime["client"]["host"]
    server_address = runtime["server"]["address"]
    perftest_config = runtime["perftest"]
    ssh_config = runtime["ssh"]
    default_duration = runtime["duration"]
    use_gpu = runtime["use_gpu"]

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "run_config.json").write_text(json.dumps(runtime, indent=2))

    server_monitor = SysMonitor(server_host, ssh_config)
    client_monitor = SysMonitor(client_host, ssh_config)

    combo_idx = 0
    result_files: list[Path] = []

    for combo in sweep_config(config):
        combo_idx += 1
        extra_args = _build_args(combo)
        extra_args_str = " ".join(extra_args)
        duration = int(combo.get("duration", default_duration))
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

        server_after = server_monitor.grab()
        client_after = client_monitor.grab()
        server_cpu_diff = SysMonitor.compute_cpu_diff(server_before, server_after)
        client_cpu_diff = SysMonitor.compute_cpu_diff(client_before, client_after)
        server_mem_info = {
            **SysMonitor.extract_mem(server_after),
            **SysMonitor.compute_mem_delta(server_before, server_after),
        }
        client_mem_info = {
            **SysMonitor.extract_mem(client_after),
            **SysMonitor.compute_mem_delta(client_before, client_after),
        }

        # Annotate with metadata
        result["_meta"] = {
            "timestamp": timestamp(),
            "test": test,
            "server": runtime["server"],
            "client": runtime["client"],
            "duration": duration,
            "parameters": combo,
            "elapsed_sec": round(elapsed, 2),
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

        if "error" in result:
            result["_meta"]["run_error"] = result["error"]

        # Write per-combo result
        combo_dir = out_path / f"{combo_idx:04d}"
        combo_dir.mkdir(parents=True, exist_ok=True)
        combo_file = combo_dir / "result.json"
        combo_file.write_text(json.dumps(result, indent=2))
        result_files.append(combo_file)

        print(f"  → {combo_file}  ({elapsed:.1f}s)", flush=True)

    # Write master summary
    summary: list[dict[str, Any]] = []
    for f in sorted(result_files):
        data = json.loads(f.read_text())
        meta = data.get("_meta", {})
        entry: dict[str, Any] = {
            "server": meta.get("server", {}),
            "client": meta.get("client", {}),
            "params": meta.get("parameters", {}),
            "error": meta.get("run_error"),
            "elapsed_sec": meta.get("elapsed_sec"),
            "cpu_per_core": meta.get("cpu_util_per_core", {}),
            "server_cpu_per_core": meta.get("server_cpu_util_per_core", {}),
            "client_cpu_per_core": meta.get("client_cpu_util_per_core", {}),
            "memory": meta.get("memory", {}),
            "server_memory": meta.get("server_memory", {}),
            "client_memory": meta.get("client_memory", {}),
        }
        entry["server_cpu_avg"] = _cpu_avg(entry["server_cpu_per_core"])
        entry["client_cpu_avg"] = _cpu_avg(entry["client_cpu_per_core"])
        entry["server_mem_used_delta"] = entry["server_memory"].get("MemUsedDelta", "")
        entry["client_mem_used_delta"] = entry["client_memory"].get("MemUsedDelta", "")
        # Copy perftest results (BW_average, MsgRate, etc.) to top level
        results = data.get("results", {})
        if isinstance(results, dict):
            for pk in ("BW_average", "MsgRate", "BW_peak", "n_iterations", "MsgSize"):
                if pk in results:
                    entry[pk] = results[pk]
        summary.append(entry)

    summary_file = out_path / "summary.json"
    summary_file.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary → {summary_file}")

    # Write CSV for quick analysis
    csv_path = out_path / "summary.csv"
    _write_csv(csv_path, summary)
    print(f"CSV     → {csv_path}")

    return result_files


def _write_csv(path: Path, summary: list[dict[str, Any]]) -> None:
    if not summary:
        return

    # Collect all keys from params and top-level
    param_keys: set[str] = set()
    for entry in summary:
        param_keys.update(entry.get("params", {}).keys())
    param_keys_sorted = sorted(k for k in param_keys if k != "report_json")

    rows: list[list[str]] = []
    bw_keys = ["BW_average", "MsgRate", "BW_peak", "n_iterations", "MsgSize"]
    # header
    headers = [
        "server_host",
        "server_address",
        "client_host",
        *param_keys_sorted,
        *bw_keys,
        "error",
        "elapsed_sec",
        "server_cpu_avg",
        "client_cpu_avg",
        "cpu_avg",
        "server_mem_used",
        "server_mem_used_delta",
        "client_mem_used",
        "client_mem_used_delta",
    ]
    rows.append(headers)

    for entry in summary:
        params = entry.get("params", {})
        server = entry.get("server", {}) or {}
        client = entry.get("client", {}) or {}
        server_cpu_avg = _cpu_avg(entry.get("server_cpu_per_core", entry.get("cpu_per_core", {})))
        client_cpu_avg = _cpu_avg(entry.get("client_cpu_per_core", {}))
        server_mem = entry.get("server_memory", entry.get("memory", {})) or {}
        client_mem = entry.get("client_memory", {}) or {}
        row = [
            str(server.get("host", "")),
            str(server.get("address", "")),
            str(client.get("host", "")),
        ]
        row += [str(params.get(k, "")) for k in param_keys_sorted]
        row += [str(entry.get(k, "")) for k in bw_keys]
        row += [
            str(entry.get("error", "") or ""),
            str(entry.get("elapsed_sec", "")),
            str(server_cpu_avg),
            str(client_cpu_avg),
            str(server_cpu_avg),
            str(server_mem.get("MemUsed", "")),
            str(server_mem.get("MemUsedDelta", "")),
            str(client_mem.get("MemUsed", "")),
            str(client_mem.get("MemUsedDelta", "")),
        ]
        rows.append(row)

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerows(rows)


def _cpu_avg(cpu_per_core: dict[str, Any]) -> float | str:
    cpu_vals = [
        float(v) for k, v in cpu_per_core.items()
        if k.startswith("cpu") and k != "cpu"
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
    memory = entry.get(key, {}) or {}
    raw = memory.get("MemUsedDelta", "")
    try:
        return parse_size(str(raw)) / (1024 * 1024)
    except (TypeError, ValueError):
        return 0.0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

SVG_COLORS = [
    "#2563eb", "#dc2626", "#16a34a", "#ca8a04",
    "#9333ea", "#0891b2", "#be123c", "#d1d5db",
]


def _svg_chart(summary: list[dict[str, Any]], report_config: dict[str, Any] | None = None) -> str:
    """Generate a static SVG report from sweep summary data."""
    report_config = report_config or {}
    qps = [e.get("params", {}).get("qp", i + 1) for i, e in enumerate(summary)]
    bw_vals = [float(e.get("BW_average", 0) or 0) for e in summary]
    rate_vals = [float(e.get("MsgRate", 0) or 0) for e in summary]

    perf_data = []
    for i in range(len(summary)):
        p = Path(summary[i].get("_result_path", f"{i+1:04d}/result.json"))
        if p.exists():
            d = json.loads(p.read_text())
            perf_data.append(d.get("_process", {}).get("server_perf", {}))
        else:
            perf_data.append({})

    all_syms = set()
    for pd in perf_data:
        for s, v in sorted(pd.items(), key=lambda x: -x[1])[:5]:
            if v > 0:
                all_syms.add(s)
    sym_total = {s: sum(pd.get(s, 0) for pd in perf_data) for s in all_syms}
    top_syms = sorted(sym_total, key=lambda s: -sym_total[s])[:7]

    series = [(sym, [pd.get(sym, 0) for pd in perf_data]) for sym in top_syms]

    cpu_key = "server_cpu_per_core" if "server_cpu_per_core" in summary[0] else "cpu_per_core"
    server_cpu_vals = [
        _as_float(e.get("server_cpu_avg", _cpu_avg(e.get(cpu_key, {}))))
        for e in summary
    ]
    client_cpu_vals = [
        _as_float(e.get("client_cpu_avg", _cpu_avg(e.get("client_cpu_per_core", {}))))
        for e in summary
    ]
    server_mem_vals = [_mem_delta_mib(e, "server_memory") for e in summary]
    client_mem_vals = [_mem_delta_mib(e, "client_memory") for e in summary]
    cores = sorted(
        [k for k in summary[0].get(cpu_key, {}) if k.startswith("cpu") and k != "cpu"],
        key=lambda c: int(c.replace("cpu", "")),
    )
    table_hdrs = ["QP"] + cores
    table_rows = [
        [str(q)] + [f'{summary[i].get(cpu_key, {}).get(c, 0):.1f}' for c in cores]
        for i, q in enumerate(qps)
    ]

    if report_config:
        title = str(report_config.get("title", "RDMA Perftest Sweep"))
        subtitle = str(report_config.get("subtitle", ""))
    else:
        first = summary[0]
        server = first.get("server", {}) or {}
        client = first.get("client", {}) or {}
        title = "RDMA Perftest Sweep"
        subtitle = (
            f"{client.get('host', 'client')} -> "
            f"{server.get('address', server.get('host', 'server'))}"
        )

    W, H, M = 1180, 1120, 16
    el: list[str] = []
    el.append(f"<svg xmlns='http://www.w3.org/2000/svg' width='{W}' height='{H}' viewBox='0 0 {W} {H}' style='background:#f8fafc'>")
    el.append(f"<text x='{W/2}' y='26' font-family='system-ui,sans-serif' font-size='18' font-weight='bold' fill='#1e293b' text-anchor='middle'>{html.escape(title)}</text>")
    if subtitle:
        el.append(f"<text x='{W/2}' y='42' font-family='system-ui,sans-serif' font-size='12' fill='#64748b' text-anchor='middle'>{html.escape(subtitle)}</text>")

    y = 56
    for col, (label, ylbl, yv, clr) in enumerate([
        ("Bandwidth (MB/s)", "MB/s", bw_vals, "#2563eb"),
        ("Message Rate (Mmsg/s)", "Kmsg/s", [r * 1000 for r in rate_vals], "#16a34a"),
    ]):
        cx = M + col * ((W - 3 * M) / 2 + M)
        cw = (W - 3 * M) / 2
        el.append(f"<rect x='{cx}' y='{y}' width='{cw}' height='220' rx='6' fill='#fff' stroke='#e2e8f0' stroke-width='1'/>")
        _line_chart(el, [float(q) for q in qps], yv, label, ylbl, cx + 8, y + 4, cw - 16, 212, clr)

    y += 220 + M
    for col, (label, ylbl, chart_series) in enumerate([
        (
            "Average CPU Utilization",
            "%",
            [
                ("server", server_cpu_vals, "#dc2626"),
                ("client", client_cpu_vals, "#0891b2"),
            ],
        ),
        (
            "Host Memory Pressure Delta",
            "MiB",
            [
                ("server", server_mem_vals, "#ca8a04"),
                ("client", client_mem_vals, "#9333ea"),
            ],
        ),
    ]):
        cx = M + col * ((W - 3 * M) / 2 + M)
        cw = (W - 3 * M) / 2
        el.append(f"<rect x='{cx}' y='{y}' width='{cw}' height='220' rx='6' fill='#fff' stroke='#e2e8f0' stroke-width='1'/>")
        _multi_line_chart(el, [float(q) for q in qps], chart_series, label, ylbl, cx + 8, y + 4, cw - 16, 212)

    y += 220 + M
    el.append(f"<rect x='{M}' y='{y}' width='{W - 2 * M}' height='250' rx='6' fill='#fff' stroke='#e2e8f0' stroke-width='1'/>")
    _stacked_bar(el, [str(q) for q in qps], series, M + 8, y + 4, W - 2 * M - 16, 242)

    y += 250 + M
    th = 30 + 24 * (len(table_rows) + 1)
    el.append(f"<rect x='{M}' y='{y}' width='{W - 2 * M}' height='{th}' rx='6' fill='#fff' stroke='#e2e8f0' stroke-width='1'/>")
    _svg_table(el, table_hdrs, table_rows, "Server Per-Core CPU Utilization (%)", M + 8, y + 4, W - 2 * M - 16)

    el.append("</svg>")
    return "\n".join(el)


def _line_chart(el: list[str], xv: list[float], yv: list[float], title: str, ylb: str, x: float, y: float, w: float, h: float, clr: str) -> None:
    pl, pr, pb, pt = 50, 20, 40, 40
    cx, cy, cw, ch = x + pl, y + pt, w - pl - pr, h - pt - pb
    ymn, ymx = 0, max(yv) * 1.1 or 1
    xmn, xmx = min(xv), max(xv)

    def px(v: float) -> float:
        return cx + (math.log2(v) - math.log2(xmn)) / (math.log2(xmx) - math.log2(xmn)) * cw if xmx != xmn else cx + cw / 2

    def py(v: float) -> float:
        return cy + ch - (v - ymn) / (ymx - ymn) * ch

    el.append(f"<text x='{x + w/2}' y='{y + 12}' font-family='system-ui,sans-serif' font-size='14' font-weight='bold' fill='#1e293b' text-anchor='middle'>{title}</text>")
    el.append(f"<text x='{x + pl/2}' y='{y + pt + ch/2}' font-family='system-ui,sans-serif' font-size='11' fill='#64748b' text-anchor='middle' transform='rotate(-90,{x + pl/2},{y + pt + ch/2})'>{ylb}</text>")
    for i in range(5):
        gy = cy + ch * i / 4
        el.append(f"<line x1='{cx}' y1='{gy}' x2='{cx + cw}' y2='{gy}' stroke='#e2e8f0' stroke-width='1'/>")
        el.append(f"<text x='{cx - 6}' y='{gy + 4}' font-family='system-ui,sans-serif' font-size='10' fill='#64748b' text-anchor='end'>{_fmt(ymn + (ymx - ymn) * (1 - i / 4), 0)}</text>")
    el.append(f"<line x1='{cx}' y1='{cy + ch}' x2='{cx + cw}' y2='{cy + ch}' stroke='#94a3b8' stroke-width='1'/>")
    for vx in xv:
        lx = px(vx)
        el.append(f"<line x1='{lx}' y1='{cy + ch}' x2='{lx}' y2='{cy + ch + 4}' stroke='#94a3b8' stroke-width='1'/>")
        el.append(f"<text x='{lx}' y='{cy + ch + 18}' font-family='system-ui,sans-serif' font-size='10' fill='#64748b' text-anchor='middle'>{int(vx)}</text>")
    pts = " ".join(f"{px(vx)},{py(vy)}" for vx, vy in zip(xv, yv))
    el.append(f"<polyline points='{pts}' fill='none' stroke='{clr}' stroke-width='2.5' stroke-linejoin='round'/>")
    for vx, vy in zip(xv, yv):
        dx, dy = px(vx), py(vy)
        el.append(f"<circle cx='{dx}' cy='{dy}' r='4' fill='{clr}'/>")
        el.append(f"<text x='{dx}' y='{dy - 10}' font-family='system-ui,sans-serif' font-size='10' fill='#1e293b' text-anchor='middle'>{_fmt(vy, 0)}</text>")


def _multi_line_chart(
    el: list[str],
    xv: list[float],
    series: list[tuple[str, list[float], str]],
    title: str,
    ylb: str,
    x: float,
    y: float,
    w: float,
    h: float,
) -> None:
    pl, pr, pb, pt = 50, 90, 40, 40
    cx, cy, cw, ch = x + pl, y + pt, w - pl - pr, h - pt - pb
    values = [v for _, vals, _ in series for v in vals]
    ymn = 0
    ymx = max(values) * 1.15 if values else 1
    if ymx <= 0:
        ymx = 1
    xmn, xmx = min(xv), max(xv)

    def px(v: float) -> float:
        return cx + (math.log2(v) - math.log2(xmn)) / (math.log2(xmx) - math.log2(xmn)) * cw if xmx != xmn else cx + cw / 2

    def py(v: float) -> float:
        return cy + ch - (v - ymn) / (ymx - ymn) * ch

    el.append(f"<text x='{x + w/2}' y='{y + 12}' font-family='system-ui,sans-serif' font-size='14' font-weight='bold' fill='#1e293b' text-anchor='middle'>{html.escape(title)}</text>")
    el.append(f"<text x='{x + pl/2}' y='{y + pt + ch/2}' font-family='system-ui,sans-serif' font-size='11' fill='#64748b' text-anchor='middle' transform='rotate(-90,{x + pl/2},{y + pt + ch/2})'>{html.escape(ylb)}</text>")
    for i in range(5):
        gy = cy + ch * i / 4
        el.append(f"<line x1='{cx}' y1='{gy}' x2='{cx + cw}' y2='{gy}' stroke='#e2e8f0' stroke-width='1'/>")
        el.append(f"<text x='{cx - 6}' y='{gy + 4}' font-family='system-ui,sans-serif' font-size='10' fill='#64748b' text-anchor='end'>{_fmt(ymn + (ymx - ymn) * (1 - i / 4), 1)}</text>")
    el.append(f"<line x1='{cx}' y1='{cy + ch}' x2='{cx + cw}' y2='{cy + ch}' stroke='#94a3b8' stroke-width='1'/>")
    for vx in xv:
        lx = px(vx)
        el.append(f"<line x1='{lx}' y1='{cy + ch}' x2='{lx}' y2='{cy + ch + 4}' stroke='#94a3b8' stroke-width='1'/>")
        el.append(f"<text x='{lx}' y='{cy + ch + 18}' font-family='system-ui,sans-serif' font-size='10' fill='#64748b' text-anchor='middle'>{int(vx)}</text>")
    for si, (name, vals, clr) in enumerate(series):
        pts = " ".join(f"{px(vx)},{py(vy)}" for vx, vy in zip(xv, vals))
        el.append(f"<polyline points='{pts}' fill='none' stroke='{clr}' stroke-width='2.5' stroke-linejoin='round'/>")
        for vx, vy in zip(xv, vals):
            el.append(f"<circle cx='{px(vx)}' cy='{py(vy)}' r='3.5' fill='{clr}'/>")
        ly = cy + 4 + si * 18
        el.append(f"<rect x='{cx + cw + 14}' y='{ly}' width='10' height='10' fill='{clr}' rx='2'/>")
        el.append(f"<text x='{cx + cw + 30}' y='{ly + 10}' font-family='system-ui,sans-serif' font-size='10' fill='#1e293b'>{html.escape(name)}</text>")


def _stacked_bar(el: list[str], xlb: list[str], series: list[tuple[str, list[float]]], x: float, y: float, w: float, h: float) -> None:
    pl, pr, pb, pt = 50, 160, 40, 40
    cx, cy, cw, ch = x + pl, y + pt, w - pl - pr, h - pt - pb
    n = len(xlb)
    bw = min(cw / n * 0.7, 50)
    gap = (cw - bw * n) / (n + 1)
    el.append(f"<text x='{x + w/2}' y='{y + 12}' font-family='system-ui,sans-serif' font-size='14' font-weight='bold' fill='#1e293b' text-anchor='middle'>Top CPU Consumers (self %)</text>")
    el.append(f"<text x='{x + pl/2}' y='{y + pt + ch/2}' font-family='system-ui,sans-serif' font-size='11' fill='#64748b' text-anchor='middle' transform='rotate(-90,{x + pl/2},{y + pt + ch/2})'>Self %</text>")
    for i in range(5):
        gy = cy + ch * i / 4
        el.append(f"<line x1='{cx}' y1='{gy}' x2='{cx + cw}' y2='{gy}' stroke='#e2e8f0' stroke-width='1'/>")
        el.append(f"<text x='{cx - 6}' y='{gy + 4}' font-family='system-ui,sans-serif' font-size='10' fill='#64748b' text-anchor='end'>{100 - 100 * i // 4}</text>")
    el.append(f"<line x1='{cx}' y1='{cy + ch}' x2='{cx + cw}' y2='{cy + ch}' stroke='#94a3b8' stroke-width='1'/>")
    for si in range(n):
        bx = cx + gap + si * (bw + gap)
        el.append(f"<text x='{bx + bw/2}' y='{cy + ch + 18}' font-family='system-ui,sans-serif' font-size='10' fill='#64748b' text-anchor='middle'>{html.escape(xlb[si])}</text>")
        bottom = 0.0
        for ci, (_, vals) in enumerate(series):
            v = vals[si]
            if v <= 0: continue
            bh = v / 100 * ch
            clr = SVG_COLORS[ci % len(SVG_COLORS)]
            el.append(f"<rect x='{bx}' y='{cy + ch - bottom - bh}' width='{bw}' height='{bh}' fill='{clr}'/>")
            bottom += v
    lx, ly2 = cx + cw + 12, cy + 4
    for ci, (name, _) in enumerate(series):
        display = name if len(name) < 32 else name[:29] + "..."
        el.append(f"<rect x='{lx}' y='{ly2}' width='10' height='10' fill='{SVG_COLORS[ci % len(SVG_COLORS)]}' rx='2'/>")
        el.append(f"<text x='{lx + 16}' y='{ly2 + 10}' font-family='system-ui,sans-serif' font-size='10' fill='#1e293b'>{html.escape(display)}</text>")
        ly2 += 18


def _svg_table(el: list[str], hdrs: list[str], rows: list[list[str]], title: str, x: float, y: float, w: float) -> None:
    ncols = len(hdrs)
    cw = w / ncols
    el.append(f"<text x='{x + w/2}' y='{y + 12}' font-family='system-ui,sans-serif' font-size='14' font-weight='bold' fill='#1e293b' text-anchor='middle'>{title}</text>")
    ty = y + 28
    for ci, hdr in enumerate(hdrs):
        el.append(f"<rect x='{x + ci * cw}' y='{ty}' width='{cw}' height='24' fill='#f1f5f9'/>")
        el.append(f"<text x='{x + ci * cw + cw/2}' y='{ty + 16}' font-family='system-ui,sans-serif' font-size='11' font-weight='bold' fill='#1e293b' text-anchor='middle'>{html.escape(hdr)}</text>")
        el.append(f"<line x1='{x + ci * cw}' y1='{ty}' x2='{x + ci * cw}' y2='{ty + 24 * (len(rows) + 1)}' stroke='#e2e8f0' stroke-width='0.5'/>")
    el.append(f"<line x1='{x + ncols * cw}' y1='{ty}' x2='{x + ncols * cw}' y2='{ty + 24 * (len(rows) + 1)}' stroke='#e2e8f0' stroke-width='0.5'/>")
    for ri, row in enumerate(rows):
        ry = ty + 24 * (ri + 1)
        bg = "#f8fafc" if ri % 2 == 1 else ""
        for ci, val in enumerate(row):
            if bg:
                el.append(f"<rect x='{x + ci * cw}' y='{ry}' width='{cw}' height='24' fill='{bg}'/>")
            el.append(f"<text x='{x + ci * cw + cw/2}' y='{ry + 16}' font-family='system-ui,sans-serif' font-size='11' font-weight='{'bold' if ci == 0 else 'normal'}' fill='#1e293b' text-anchor='middle'>{html.escape(val)}</text>")


def _fmt(v: float, d: int = 0) -> str:
    return f"{v:.0f}" if v >= 1000 else f"{v:.{d}f}"


def generate_report(output_dir: str) -> None:
    """Generate SVG (and optionally PDF) report from existing sweep results."""
    out = Path(output_dir)
    summary = json.loads((out / "summary.json").read_text())
    if not summary:
        raise ValueError("summary.json contains no sweep entries")
    for i in range(len(summary)):
        summary[i]["_result_path"] = str(out / f"{i+1:04d}" / "result.json")
    report_config: dict[str, Any] = {}
    run_config_path = out / "run_config.json"
    if run_config_path.exists():
        report_config = json.loads(run_config_path.read_text()).get("report", {})
    svg = _svg_chart(summary, report_config=report_config)
    svg_path = out / "chart.svg"
    svg_path.write_text(svg)
    print(f"Report SVG → {svg_path}")
    try:
        import subprocess
        r = subprocess.run(["which", "cairosvg"], capture_output=True, text=True)
        if r.returncode == 0:
            pdf_path = out / "chart.pdf"
            subprocess.run(["cairosvg", str(svg_path), "-o", str(pdf_path)], check=True)
            print(f"Report PDF → {pdf_path}")
    except Exception:
        pass


def main() -> None:
    ap = argparse.ArgumentParser(
        description="RDMA perftest parameter sweep tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--config", "-c", help="YAML sweep config (run sweep mode)")
    ap.add_argument("--output-dir", "-o", default="sweep_results", help="Output directory")
    ap.add_argument("--report", "-r", nargs="?", const=True, default=False, help="Generate report from existing results (optional: path to results dir)")
    args = ap.parse_args()

    report_dir = args.report if isinstance(args.report, str) else args.output_dir
    if args.report:
        generate_report(report_dir)
        return

    if yaml is None:
        print("ERROR: PyYAML is required.  pip install pyyaml", file=sys.stderr)
        sys.exit(1)

    if not args.config:
        ap.print_help()
        sys.exit(1)

    config = yaml.safe_load(Path(args.config).read_text())

    if not config.get("sweep"):
        print("ERROR: config must contain a 'sweep' list", file=sys.stderr)
        sys.exit(1)

    run_sweep(config, output_dir=args.output_dir)


# Pre-populate local host set so all components can detect self immediately.
init_local_hosts()

if __name__ == "__main__":
    main()
