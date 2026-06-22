"""Configuration normalization for rdma_sweep."""

from __future__ import annotations

import ipaddress
import shlex
from typing import Any

DEFAULT_PERFTEST_CONFIG: dict[str, Any] = {
    "dir": "",
    "rdma_core_lib": "",
    "env": {},
    "tmp_dir": "/tmp/rdma_sweep_{run_id}",
    "json_file": "{tmp_dir}/perftest_out.json",
    "time_file": "{tmp_dir}/perftest_time.out",
    "server_pid_file": "{tmp_dir}/perftest_server.pid",
    "server_log_file": "{tmp_dir}/perftest_server.log",
    "perf_data": "{tmp_dir}/perftest_perf.data",
    "perf_pid_file": "{tmp_dir}/perftest_perf.pid",
    "perf_record": True,
    "wait_timeout": 30,
    "default_port": 18515,
}

DEFAULT_SSH_CONFIG: dict[str, Any] = {
    "sudo": True,
    "connect_timeout": 10,
    "options": [
        "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=accept-new",
    ],
}

PATH_KEYS = (
    "json_file",
    "time_file",
    "server_pid_file",
    "server_log_file",
    "perf_data",
    "perf_pid_file",
)


def parse_bool(value: Any, key: str) -> bool:
    """Parse config booleans strictly enough to catch accidental strings."""
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "on", "1"}:
            return True
        if normalized in {"false", "no", "off", "0"}:
            return False
    raise ValueError(f"{key} must be a boolean")


def deep_merge(base: dict[str, Any], override: dict[str, Any] | None) -> dict[str, Any]:
    """Return ``base`` recursively merged with ``override``."""
    result = dict(base)
    if not override:
        return result
    for key, value in override.items():
        if isinstance(result.get(key), dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def strip_user(host: str) -> str:
    """Turn an SSH target such as user@host into host for comparisons."""
    return host.rsplit("@", 1)[-1].strip()


def is_loopback(value: str) -> bool:
    host = strip_user(value).strip().lower()
    if host in {"localhost", "ip6-localhost"}:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def endpoint_config(config: dict[str, Any], name: str) -> dict[str, Any]:
    raw = config.get(name)
    if isinstance(raw, str):
        endpoint: dict[str, Any] = {"host": raw}
    elif isinstance(raw, dict):
        endpoint = dict(raw)
    elif raw is None:
        endpoint = {}
    else:
        raise ValueError(f"config key '{name}' must be a mapping or string")

    if name == "server":
        endpoint.setdefault("host", config.get("server_host", ""))
        endpoint.setdefault(
            "address",
            config.get("server_address", config.get("server_addr", "")),
        )
    elif name == "client":
        endpoint.setdefault("host", config.get("client_host", ""))

    if "rdma_address" in endpoint and "address" not in endpoint:
        endpoint["address"] = endpoint["rdma_address"]
    return endpoint


def runtime_config(config: dict[str, Any]) -> dict[str, Any]:
    """Normalize YAML config into explicit server/client/runtime sections."""
    server = endpoint_config(config, "server")
    client = endpoint_config(config, "client")

    server_host = str(server.get("host", "")).strip()
    client_host = str(client.get("host", "")).strip()
    if not server_host:
        raise ValueError("config must set server.host (or legacy server_host)")
    if not client_host:
        raise ValueError("config must set client.host (or legacy client_host)")
    if strip_user(server_host).lower() == strip_user(client_host).lower():
        raise ValueError("server.host and client.host must be different machines")

    server_address = str(server.get("address", "")).strip()
    if not server_address:
        raise ValueError("config must set server.address to the server RDMA address")
    if is_loopback(server_address):
        raise ValueError("server.address must be reachable by the client, not loopback")
    server["host"] = server_host
    server["address"] = server_address
    client["host"] = client_host

    perftest_override = dict(config.get("perftest", {}) or {})
    if "perftest_dir" in config:
        perftest_override.setdefault("dir", config["perftest_dir"])
    if "rdma_core_lib" in config:
        perftest_override.setdefault("rdma_core_lib", config["rdma_core_lib"])
    perftest = deep_merge(DEFAULT_PERFTEST_CONFIG, perftest_override)
    perftest_dir = str(perftest.get("dir", "")).strip()
    if not perftest_dir:
        raise ValueError("config must set perftest.dir")
    perftest["dir"] = perftest_dir
    if not isinstance(perftest.get("env"), dict):
        raise ValueError("perftest.env must be a mapping")
    perftest["wait_timeout"] = int(perftest.get("wait_timeout", 30))
    perftest["default_port"] = int(perftest.get("default_port", 18515))
    perftest["perf_record"] = parse_bool(perftest.get("perf_record", True), "perftest.perf_record")

    ssh = deep_merge(DEFAULT_SSH_CONFIG, config.get("ssh", {}) or {})
    ssh["sudo"] = parse_bool(ssh.get("sudo", True), "ssh.sudo")
    ssh["connect_timeout"] = int(ssh.get("connect_timeout", 10))
    if isinstance(ssh.get("options"), str):
        ssh["options"] = shlex.split(str(ssh["options"]))
    else:
        ssh["options"] = [str(opt) for opt in ssh.get("options", [])]

    test = str(config.get("test", "ib_write_bw"))
    perf_note = "server perf record -g" if perftest["perf_record"] else "perf record disabled"
    report = deep_merge(
        {
            "title": "RDMA Perftest Sweep",
            "subtitle": f"{client_host} -> {server_address} - {test} - {perf_note}",
        },
        config.get("report", {}) or {},
    )

    return {
        "test": test,
        "duration": int(config.get("duration", 10)),
        "use_gpu": parse_bool(config.get("use_gpu", False), "use_gpu"),
        "server": server,
        "client": client,
        "perftest": perftest,
        "ssh": ssh,
        "report": report,
    }


def resolve_perftest_paths(perftest_config: dict[str, Any], run_id: str) -> dict[str, Any]:
    """Resolve per-run path templates such as ``{tmp_dir}`` and ``{run_id}``."""
    resolved = dict(perftest_config)
    tmp_dir = str(resolved.get("tmp_dir", "/tmp/rdma_sweep_{run_id}"))
    tmp_dir = tmp_dir.replace("{run_id}", run_id)
    resolved["tmp_dir"] = tmp_dir

    mapping = {"{run_id}": run_id, "{tmp_dir}": tmp_dir}
    for key in PATH_KEYS:
        value = str(resolved[key])
        for token, replacement in mapping.items():
            value = value.replace(token, replacement)
        resolved[key] = value
    return resolved
