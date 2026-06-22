"""Remote command helpers for rdma_sweep."""

from __future__ import annotations

import subprocess
import shlex
from dataclasses import dataclass
from typing import Any

from rdma_config import DEFAULT_SSH_CONFIG, deep_merge, parse_bool, strip_user

_LOCAL_HOSTS: set[str] = set()


def _as_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value


@dataclass
class RemoteResult:
    host: str
    command: str
    stdout: str = ""
    stderr: str = ""
    returncode: int = 0
    exception: str = ""
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.exception and not self.timed_out

    def error_summary(self) -> str:
        if self.timed_out:
            return f"timed out on {self.host}: {self.command}"
        if self.exception:
            return self.exception
        if self.stderr.strip():
            return self.stderr.strip()
        return f"exit {self.returncode}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "host": self.host,
            "command": self.command,
            "returncode": self.returncode,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "exception": self.exception,
            "timed_out": self.timed_out,
        }


def init_local_hosts() -> None:
    """Populate the local-host set with this machine's hostnames/IPs."""
    hosts = {"127.0.0.1", "localhost", "::1"}
    try:
        out = subprocess.check_output(
            ["hostname", "-A"], text=True, timeout=5, stderr=subprocess.DEVNULL,
        )
        for host in out.split():
            hosts.add(host.strip())
    except Exception:
        pass
    try:
        out = subprocess.check_output(
            ["hostname", "-I"], text=True, timeout=5, stderr=subprocess.DEVNULL,
        )
        for host in out.split():
            hosts.add(host.strip())
    except Exception:
        pass
    _LOCAL_HOSTS.clear()
    _LOCAL_HOSTS.update(hosts)


def run_local_result(
    cmd: str,
    timeout: int = 300,
    sudo: bool = True,
    host: str = "localhost",
) -> RemoteResult:
    argv = ["sudo", "bash", "-c", cmd] if sudo else ["bash", "-c", cmd]
    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return RemoteResult(
            host=host,
            command=cmd,
            stdout=proc.stdout,
            stderr=proc.stderr,
            returncode=proc.returncode,
        )
    except subprocess.TimeoutExpired as exc:
        return RemoteResult(
            host=host,
            command=cmd,
            stdout=_as_text(exc.stdout),
            stderr=_as_text(exc.stderr),
            returncode=124,
            timed_out=True,
        )
    except Exception as exc:
        return RemoteResult(host=host, command=cmd, returncode=1, exception=str(exc))


def run_remote_result(
    cmd: str,
    host: str,
    timeout: int = 300,
    ssh_config: dict[str, Any] | None = None,
    sudo: bool | None = None,
) -> RemoteResult:
    """Run a command locally or through SSH and preserve failure evidence."""
    effective_ssh = deep_merge(DEFAULT_SSH_CONFIG, ssh_config or {})
    sudo_enabled = parse_bool(effective_ssh.get("sudo", True), "ssh.sudo") if sudo is None else sudo

    if not _LOCAL_HOSTS:
        init_local_hosts()

    allow_local = parse_bool(effective_ssh.get("allow_local", False), "ssh.allow_local")
    if allow_local and (host in _LOCAL_HOSTS or strip_user(host) in _LOCAL_HOSTS):
        return run_local_result(cmd, timeout=timeout, sudo=sudo_enabled, host=host)

    safe = shlex.join(["bash", "-c", cmd])
    remote_cmd = f"sudo {safe}" if sudo_enabled else safe
    ssh_args = [
        "ssh",
        "-o", f"ConnectTimeout={int(effective_ssh.get('connect_timeout', 10))}",
        *[str(opt) for opt in effective_ssh.get("options", [])],
        host,
        remote_cmd,
    ]
    try:
        proc = subprocess.run(
            ssh_args,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return RemoteResult(
            host=host,
            command=cmd,
            stdout=proc.stdout,
            stderr=proc.stderr,
            returncode=proc.returncode,
        )
    except subprocess.TimeoutExpired as exc:
        return RemoteResult(
            host=host,
            command=cmd,
            stdout=_as_text(exc.stdout),
            stderr=_as_text(exc.stderr),
            returncode=124,
            timed_out=True,
        )
    except Exception as exc:
        return RemoteResult(host=host, command=cmd, returncode=1, exception=str(exc))


def run_remote(
    cmd: str,
    host: str,
    timeout: int = 300,
    check: bool = True,
    ssh_config: dict[str, Any] | None = None,
    sudo: bool | None = None,
) -> str:
    result = run_remote_result(
        cmd,
        host,
        timeout=timeout,
        ssh_config=ssh_config,
        sudo=sudo,
    )
    if check and not result.ok:
        raise subprocess.CalledProcessError(
            result.returncode,
            cmd,
            result.stdout,
            result.stderr or result.error_summary(),
        )
    return result.stdout
