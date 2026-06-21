#!/usr/bin/env python3
"""RDMA parameter sweep tool for perftest.

Operations:
  1. rdma_sweep.py -c config.yaml -o results/     # run sweep
  2. rdma_sweep.py --report results/               # generate SVG report
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]


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
    for unit in ("B", "K", "M", "G"):
        if n < 1024:
            return f"{n:.1f}{unit}" if unit != "B" else f"{n}{unit}"
        n /= 1024
    return f"{n:.1f}T"


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

    def __init__(self, host: str) -> None:
        self.host = host

    def _run(self, cmd: str) -> list[str]:
        if self.host in _LOCAL_HOSTS:
            return subprocess.check_output(
                cmd, shell=True, text=True, timeout=10,
            ).splitlines()
        return subprocess.check_output(
            ["ssh", "-o", "ConnectTimeout=3", "-o", "StrictHostKeyChecking=no",
             "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR",
             self.host, cmd],
            text=True, timeout=10,
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
        # ponytail: only pull the lines we actually use
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
        # ponytail: MemUsed delta is what matters for perftest
        def _used(m: dict[str, int]) -> int:
            return m.get("MemTotal", 0) - m.get("MemFree", 0) - m.get("Buffers", 0) - m.get("Cached", 0)
        used_delta_kb = _used(a) - _used(b)
        return {
            "MemUsedDelta": format_size(abs(used_delta_kb) * 1024),
            "MemFreeDelta": format_size((a.get("MemFree", 0) - b.get("MemFree", 0)) * 1024),
        }


# ---------------------------------------------------------------------------
# Perftest runner
# ---------------------------------------------------------------------------

PERFTEST_BINS = [
    "ib_write_bw", "ib_read_bw", "ib_send_bw",
    "ib_write_lat", "ib_read_lat", "ib_send_lat",
]

# Remote environment config
REMOTE_HOME = "/home/dpu"
PERFTEST_BIN = "/tmp/perftest/ib_write_bw"
RDMA_CORE_LIB = "/tmp/rdma-core/build/lib"
JSON_FILE = "perftest_out.json"


_LOCAL_HOSTS: set[str] = set()

# --- initialise at bottom of file, after _init_local_hosts is defined ---


def _init_local_hosts() -> None:
    """Populate _LOCAL_HOSTS with this machine's own hostnames/IPs."""
    global _LOCAL_HOSTS
    hosts = {"127.0.0.1", "localhost", "::1"}
    try:
        out = subprocess.check_output(
            ["hostname", "-A"], text=True, timeout=5, stderr=subprocess.DEVNULL,
        )
        for h in out.split():
            hosts.add(h.strip())
    except Exception:
        pass
    try:
        out = subprocess.check_output(
            ["hostname", "-I"], text=True, timeout=5, stderr=subprocess.DEVNULL,
        )
        for h in out.split():
            hosts.add(h.strip())
    except Exception:
        pass
    _LOCAL_HOSTS = hosts


def _local_sudo(cmd: str, timeout: int = 300) -> str:
    """Run a command with sudo directly (no SSH)."""
    proc = subprocess.run(
        ["sudo", "bash", "-c", cmd],
        capture_output=True, text=True, timeout=timeout,
    )
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(
            proc.returncode, cmd, proc.stdout, proc.stderr
        )
    return proc.stdout


def _ssh(cmd: str, host: str, timeout: int = 300, check: bool = True) -> str:
    """Run a command via SSH with sudo, return stdout.

    If *host* is this machine (any IP/hostname), runs locally to avoid
    SSH-key-dance on benchmarking machines.
    """
    if not _LOCAL_HOSTS:
        _init_local_hosts()

    if host in _LOCAL_HOSTS:
        try:
            return _local_sudo(cmd, timeout=timeout)
        except Exception as exc:
            if check:
                raise
            return ""

    try:
        # Wrap entire cmd under sudo so compound commands (separated by ; or
        # containing &/redirects) all run with the same privilege level.
        safe = shlex.quote(cmd)
        proc = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=10", "-o", "StrictHostKeyChecking=no",
             "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR",
             host, f"sudo bash -c {safe}"],
            capture_output=True, text=True, timeout=timeout,
        )
        if proc.returncode != 0:
            raise subprocess.CalledProcessError(
                proc.returncode, cmd, proc.stdout, proc.stderr
            )
        return proc.stdout
    except Exception as exc:
        if check:
            raise
        return ""


def _wait_for_port(host: str, port: int, timeout: int = 30) -> None:
    """Poll *host* until *port* is listening (or *timeout* seconds elapse)."""
    for _ in range(timeout * 2):
        out = _ssh(
            f"ss -tlnp 2>/dev/null | grep -q ':{port}' && echo ready",
            host, check=False,
        )
        if "ready" in out:
            return
        time.sleep(0.5)


PERF_DATA = "/tmp/perftest_perf.data"
PERF_PID_FILE = "/tmp/perftest_perf.pid"


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
    perftest_dir: str,
    extra_args: list[str],
    duration: int,
    use_gpu: bool,
) -> dict[str, Any]:
    """Run one perftest measurement and return parsed JSON + metadata.

    Launches the server in the background over SSH, runs the client with
    ``--out_json`` (writes JSON to a file), reads the file back, then
    stops the server.  Server-side CPU attribution via ``perf record -g``,
    client-side resource accounting via ``/usr/bin/time``.
    """
    bin_path = os.path.join(perftest_dir, binary) if perftest_dir else f"/tmp/perftest/{binary}"
    if use_gpu:
        bin_path += "_gpu"

    json_file = f"/tmp/{JSON_FILE}"
    env = f"LD_LIBRARY_PATH={RDMA_CORE_LIB}"

    # Build args: strip out any --out_json (we add our own), add -D for duration
    filtered = [a for a in extra_args
                if a not in ("--out_json",) and not a.startswith("--out_json_file")]
    args_str = " ".join(shlex.quote(a) for a in filtered)

    # Both server and client need the same -D, -s, -q, -t, -r args
    server_cmd = f"{env} {bin_path} --server {args_str} -D {duration}"

    try:
        _ssh(
            f"rm -f /tmp/perftest_server.pid; "
            f"{server_cmd} &>/tmp/perftest_server.log & echo $! > /tmp/perftest_server.pid",
            server_host, check=False,
        )
        port = 18515
        for i in range(0, len(extra_args) - 1):
            if extra_args[i] == "-p":
                port = int(extra_args[i + 1])
                break
        _wait_for_port(server_host, port)

        # Capture server PID
        server_pid_raw = _ssh("cat /tmp/perftest_server.pid", server_host, check=False).strip()
        server_pid = int(server_pid_raw) if server_pid_raw and server_pid_raw.isdecimal() else None

        # Start perf record -g on server PID (callchain sampling)
        perf_started = False
        if server_pid:
            _ssh(
                f"sudo rm -f {PERF_DATA} {PERF_PID_FILE}; "
                f"sudo perf record -g -p {server_pid} -F 99 -o {PERF_DATA} "
                f">/dev/null 2>&1 & echo $! > {PERF_PID_FILE}",
                server_host, check=False,
            )
            perf_started = True

        # Run client wrapped with /usr/bin/time for process resource accounting
        # perftest often returns exit code 1 even on success (cosmetic/non-fatal
        # warnings) so we tolerate non-zero exit.
        time_fmt = "%U %S %P %M %c %w"
        time_file = "/tmp/perftest_time.out"
        _ssh(
            f"rm -f {json_file} {time_file}; "
            f"LD_LIBRARY_PATH={RDMA_CORE_LIB} /usr/bin/time --format='{time_fmt}' "
            f"{bin_path} 127.0.0.1 {args_str} -D {duration} "
            f"--out_json --out_json_file={json_file} 2>{time_file}",
            server_host, check=False, timeout=duration + 60,
        )

        # Parse /usr/bin/time output (client-side user/sys, RSS)
        proc_usage: dict[str, Any] = {}
        time_raw = _ssh(f"cat {time_file} 2>/dev/null || true", server_host, check=False).strip()
        if time_raw:
            parts = time_raw.split()
            if len(parts) >= 4:
                proc_usage = {
                    "client_user_sec": parts[0],
                    "client_sys_sec": parts[1],
                    "client_cpu_pct": parts[2].rstrip("%"),
                    "client_max_rss_kb": parts[3],
                }

        # Stop perf record (SIGINT flushes data and exits gracefully)
        perf_report: dict[str, float] = {}
        if perf_started:
            _ssh(
                f"ppid=$(cat {PERF_PID_FILE} 2>/dev/null) && "
                f"sudo kill -INT $ppid 2>/dev/null; "
                f"sleep 2; "  # wait for perf to finalise perf.data
                f"sudo rm -f {PERF_PID_FILE}",
                server_host, check=False, timeout=15,
            )
            report_raw = _ssh(
                f"sudo perf report --stdio --no-header -g none -i {PERF_DATA} 2>/dev/null || true",
                server_host, check=False,
            )
            _ssh(f"sudo rm -f {PERF_DATA}", server_host, check=False)
            perf_report = _parse_perf_report(report_raw)

        # Read JSON result
        result_raw = _ssh(f"cat {json_file} 2>/dev/null || true", server_host)
        result = _parse_json_output(result_raw)
        result["_process"] = {
            "server_pid": server_pid,
            "server_perf": perf_report,
            "client_usage": proc_usage,
        }
        return result
    except Exception as exc:
        return {"error": str(exc)}
    finally:
        _cancel(server_host, bin_path)


def _cancel(host: str, bin_name: str) -> None:
    try:
        pid = _ssh("cat /tmp/perftest_server.pid 2>/dev/null || true", host, check=False).strip()
        if pid:
            _ssh(f"kill -9 {pid} 2>/dev/null || true; rm -f /tmp/perftest_server.pid", host)
    except Exception:
        pass


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
          server_host: dpu
          perftest_dir: /path
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
        "duration": "-D",
        "recv_post_list": "--recv-post-list",
        "cpu_util": "--cpu_util",
        "device": "-d",
        "check_alive": "--check-alive",
    }
    skip = {"perftest_dir", "server_host", "test", "use_gpu", "output_dir", "host"}
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
    test = config.get("test", "ib_write_bw")
    server_host = config.get("server_host", "")
    perftest_dir = config.get("perftest_dir", "/root/perftest")
    duration = config.get("duration", 10)
    use_gpu = config.get("use_gpu", False)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    monitor = SysMonitor(server_host)

    combo_idx = 0
    result_files: list[Path] = []

    for combo in sweep_config(config):
        combo_idx += 1
        extra_args = _build_args(combo)
        extra_args_str = " ".join(extra_args)
        label = f"[{combo_idx}] {test} msg={combo.get('msg_size','?')} qp={combo.get('qp','?')}"

        print(f"{timestamp()}  {label}  args: {extra_args_str}", flush=True)

        # Sample sys state before test
        sys_before = monitor.grab()

        t0 = time.monotonic()
        result = run_perftest(
            binary=test,
            server_host=server_host,
            perftest_dir=perftest_dir,
            extra_args=extra_args,
            duration=duration,
            use_gpu=use_gpu,
        )
        elapsed = time.monotonic() - t0

        sys_after = monitor.grab()
        cpu_diff = SysMonitor.compute_cpu_diff(sys_before, sys_after)
        mem_after = SysMonitor.extract_mem(sys_after)
        mem_delta = SysMonitor.compute_mem_delta(sys_before, sys_after)
        mem_info = {**mem_after, **mem_delta}

        # Annotate with metadata
        result["_meta"] = {
            "timestamp": timestamp(),
            "test": test,
            "parameters": combo,
            "elapsed_sec": round(elapsed, 2),
            "cpu_util_per_core": cpu_diff,
            "memory": mem_info,
            "sys_before_ok": "error" not in sys_before,
            "sys_after_ok": "error" not in sys_after,
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
            "params": meta.get("parameters", {}),
            "error": meta.get("run_error"),
            "elapsed_sec": meta.get("elapsed_sec"),
            "cpu_per_core": meta.get("cpu_util_per_core", {}),
            "memory": meta.get("memory", {}),
        }
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


def _write_csv(path: Path, summary: list[dict]) -> None:
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
    headers = param_keys_sorted + bw_keys + ["error", "elapsed_sec", "cpu_avg", "mem_used", "mem_used_delta"]
    rows.append(headers)

    for entry in summary:
        params = entry.get("params", {})
        cpu_vals = [
            v for k, v in entry.get("cpu_per_core", {}).items()
            if k.startswith("cpu") and k != "cpu"
        ]
        cpu_avg = round(sum(cpu_vals) / len(cpu_vals), 1) if cpu_vals else ""
        mem = entry.get("memory", {})
        mem_used = mem.get("MemUsed", "")
        mem_delta = mem.get("MemUsedDelta", "")
        row = [str(params.get(k, "")) for k in param_keys_sorted]
        row += [str(entry.get(k, "")) for k in bw_keys]
        row += [
            str(entry.get("error", "") or ""),
            str(entry.get("elapsed_sec", "")),
            str(cpu_avg),
            str(mem_used),
            str(mem_delta),
        ]
        rows.append(row)

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerows(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

SVG_COLORS = [
    "#2563eb", "#dc2626", "#16a34a", "#ca8a04",
    "#9333ea", "#0891b2", "#be123c", "#d1d5db",
]


def _svg_chart(summary: list[dict]) -> str:
    """Generate a static SVG report from sweep summary data."""
    qps = [e["params"]["qp"] for e in summary]
    bw_vals = [e["BW_average"] for e in summary]
    rate_vals = [e["MsgRate"] for e in summary]

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

    cores = sorted(
        [k for k in summary[0].get("cpu_per_core", {}) if k.startswith("cpu") and k != "cpu"],
        key=lambda c: int(c.replace("cpu", "")),
    )
    table_hdrs = ["QP"] + cores
    table_rows = [
        [str(q)] + [f'{summary[i]["cpu_per_core"].get(c, 0):.1f}' for c in cores]
        for i, q in enumerate(qps)
    ]

    W, H, M = 1100, 900, 16
    el: list[str] = []
    el.append(f"<svg xmlns='http://www.w3.org/2000/svg' width='{W}' height='{H}' viewBox='0 0 {W} {H}' style='background:#f8fafc'>")
    el.append(f"<text x='{W/2}' y='26' font-family='system-ui,sans-serif' font-size='18' font-weight='bold' fill='#1e293b' text-anchor='middle'>RDMA Write BW Sweep</text>")
    el.append(f"<text x='{W/2}' y='42' font-family='system-ui,sans-serif' font-size='12' fill='#64748b' text-anchor='middle'>SoftRoCE (rxe0) &#183; 64K msg &#183; ib_write_bw &#183; server perf record -g</text>")

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
    el.append(f"<rect x='{M}' y='{y}' width='{W - 2 * M}' height='280' rx='6' fill='#fff' stroke='#e2e8f0' stroke-width='1'/>")
    _stacked_bar(el, [str(q) for q in qps], series, M + 8, y + 4, W - 2 * M - 16, 272)

    y += 280 + M
    th = 30 + 24 * (len(table_rows) + 1)
    el.append(f"<rect x='{M}' y='{y}' width='{W - 2 * M}' height='{th}' rx='6' fill='#fff' stroke='#e2e8f0' stroke-width='1'/>")
    _svg_table(el, table_hdrs, table_rows, "Per-Core CPU Utilization (%)", M + 8, y + 4, W - 2 * M - 16)

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
        el.append(f"<text x='{bx + bw/2}' y='{cy + ch + 18}' font-family='system-ui,sans-serif' font-size='10' fill='#64748b' text-anchor='middle'>{xlb[si]}</text>")
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
        el.append(f"<text x='{lx + 16}' y='{ly2 + 10}' font-family='system-ui,sans-serif' font-size='10' fill='#1e293b'>{display}</text>")
        ly2 += 18


def _svg_table(el: list[str], hdrs: list[str], rows: list[list[str]], title: str, x: float, y: float, w: float) -> None:
    ncols = len(hdrs)
    cw = w / ncols
    el.append(f"<text x='{x + w/2}' y='{y + 12}' font-family='system-ui,sans-serif' font-size='14' font-weight='bold' fill='#1e293b' text-anchor='middle'>{title}</text>")
    ty = y + 28
    for ci, hdr in enumerate(hdrs):
        el.append(f"<rect x='{x + ci * cw}' y='{ty}' width='{cw}' height='24' fill='#f1f5f9'/>")
        el.append(f"<text x='{x + ci * cw + cw/2}' y='{ty + 16}' font-family='system-ui,sans-serif' font-size='11' font-weight='bold' fill='#1e293b' text-anchor='middle'>{hdr}</text>")
        el.append(f"<line x1='{x + ci * cw}' y1='{ty}' x2='{x + ci * cw}' y2='{ty + 24 * (len(rows) + 1)}' stroke='#e2e8f0' stroke-width='0.5'/>")
    el.append(f"<line x1='{x + ncols * cw}' y1='{ty}' x2='{x + ncols * cw}' y2='{ty + 24 * (len(rows) + 1)}' stroke='#e2e8f0' stroke-width='0.5'/>")
    for ri, row in enumerate(rows):
        ry = ty + 24 * (ri + 1)
        bg = "#f8fafc" if ri % 2 == 1 else ""
        for ci, val in enumerate(row):
            if bg:
                el.append(f"<rect x='{x + ci * cw}' y='{ry}' width='{cw}' height='24' fill='{bg}'/>")
            el.append(f"<text x='{x + ci * cw + cw/2}' y='{ry + 16}' font-family='system-ui,sans-serif' font-size='11' font-weight='{'bold' if ci == 0 else 'normal'}' fill='#1e293b' text-anchor='middle'>{val}</text>")


def _fmt(v: float, d: int = 0) -> str:
    return f"{v:.0f}" if v >= 1000 else f"{v:.{d}f}"


def generate_report(output_dir: str) -> None:
    """Generate SVG (and optionally PDF) report from existing sweep results."""
    out = Path(output_dir)
    summary = json.loads((out / "summary.json").read_text())
    for i in range(len(summary)):
        summary[i]["_result_path"] = str(out / f"{i+1:04d}" / "result.json")
    svg = _svg_chart(summary)
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
_init_local_hosts()

if __name__ == "__main__":
    main()
