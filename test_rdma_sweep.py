"""Tests for rdma_sweep — unit-testable components.

Run: python3 -m pytest rdma_sweep_tool/test_rdma_sweep.py -v
Or:  python3 -m rdma_sweep_tool.test_rdma_sweep
"""

import unittest
import sys
import os
import json
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(__file__))

from rdma_sweep import (
    DEFAULT_PERFTEST_CONFIG,
    SysMonitor,
    _build_args,
    _cancel,
    _parse_json_output,
    _parse_perf_report,
    _runtime_config,
    _validate_perftest_metrics,
    generate_report,
    run_perftest,
)
from rdma_remote import RemoteResult, run_local_result, run_remote_result


# ---------------------------------------------------------------------------
# _parse_perf_report  (12 scenarios)
# ---------------------------------------------------------------------------

class TestParsePerfReport(unittest.TestCase):
    """_parse_perf_report parses ``perf report --stdio --no-header -g none``."""

    def test_empty_string(self):
        self.assertEqual(_parse_perf_report(""), {})

    def test_only_comments(self):
        raw = (
            "# To display the perf.data header info, please use --header/--header-only options.\n"
            "#\n"
            "# Total Lost Samples: 0\n"
            "#\n"
            "# Overhead  Command      Shared Object      Symbol\n"
            "# ........  ...........  .................  ...........................\n"
        )
        self.assertEqual(_parse_perf_report(raw), {})

    def test_single_userspace_symbol(self):
        raw = "    35.20%    35.20%  ib_write_bw  ib_write_bw  [.] run_iterations\n"
        self.assertEqual(_parse_perf_report(raw), {"run_iterations@ib_write_bw": 35.2})

    def test_single_kernel_symbol(self):
        raw = "     8.50%     8.50%  ib_write_bw  [kernel.kallsyms]  [k] rxe_requester\n"
        self.assertEqual(_parse_perf_report(raw), {"rxe_requester": 8.5})

    def test_multiple_symbols_sorted(self):
        raw = (
            "    35.20%    35.20%  ib_write_bw  ib_write_bw        [.] run_iterations\n"
            "    22.15%    22.15%  ib_write_bw  libibverbs.so.1   [.] ibv_post_send\n"
            "    12.30%    12.30%  ib_write_bw  libibverbs.so.1   [.] ibv_poll_cq\n"
            "     8.50%     8.50%  ib_write_bw  [kernel.kallsyms]  [k] rxe_requester\n"
            "     3.90%     3.90%  ib_write_bw  [kernel.kallsyms]  [k] rxe_send\n"
        )
        expected = {
            "run_iterations@ib_write_bw": 35.2,
            "ibv_post_send@libibverbs.so.1": 22.15,
            "ibv_poll_cq@libibverbs.so.1": 12.3,
            "rxe_requester": 8.5,
            "rxe_send": 3.9,
        }
        self.assertEqual(_parse_perf_report(raw), expected)

    def test_skips_zero_self(self):
        raw = (
            "    95.52%     0.00%  ib_write_bw  ib_write_bw  [.] _start\n"
            "     3.78%     3.78%  ib_write_bw  [kernel.kallsyms]  [k] eth_type_trans\n"
            "     0.70%     0.70%  ib_write_bw  [kernel.kallsyms]  [k] pvclock_clocksource_read_nowd\n"
        )
        self.assertEqual(_parse_perf_report(raw), {
            "eth_type_trans": 3.78,
            "pvclock_clocksource_read_nowd": 0.7,
        })

    def test_skips_lines_without_annotation(self):
        raw = (
            "    35.20%    35.20%  ib_write_bw  ib_write_bw  [.] run_iterations\n"
            "            |\n"
            "            ---0\n"
            "               ibv_post_send\n"
            "               ibv_poll_cq\n"
        )
        self.assertEqual(_parse_perf_report(raw), {"run_iterations@ib_write_bw": 35.2})

    def test_skips_malformed_lines(self):
        raw = (
            "    35.20%    35.20%  ib_write_bw  ib_write_bw  [.] run_iterations\n"
            "short_line\n"
            "    22.15%    22.15%  [.] ibv_post_send\n"
        )
        self.assertEqual(_parse_perf_report(raw), {"run_iterations@ib_write_bw": 35.2})

    def test_full_realistic_output(self):
        raw = (
            "    35.20%    35.20%  ib_write_bw  ib_write_bw        [.] run_iterations\n"
            "    22.15%    22.15%  ib_write_bw  libibverbs.so.1   [.] ibv_post_send\n"
            "    12.30%    12.30%  ib_write_bw  libibverbs.so.1   [.] ibv_poll_cq\n"
            "    10.00%     0.00%  ib_write_bw  ib_write_bw  [.] _start\n"
            "     8.50%     8.50%  ib_write_bw  [kernel.kallsyms]  [k] rxe_requester\n"
            "     5.00%     0.00%  ib_write_bw  libibverbs.so.1  [.] ibv_create_cq\n"
            "     3.90%     3.90%  ib_write_bw  [kernel.kallsyms]  [k] rxe_send\n"
            "     1.20%     1.20%  ib_write_bw  librxe-rdmav59.so  [.] rxe_create_cq_ex\n"
        )
        expected = {
            "run_iterations@ib_write_bw": 35.2,
            "ibv_post_send@libibverbs.so.1": 22.15,
            "ibv_poll_cq@libibverbs.so.1": 12.3,
            "rxe_requester": 8.5,
            "rxe_send": 3.9,
            "rxe_create_cq_ex@librxe-rdmav59.so": 1.2,
        }
        self.assertEqual(_parse_perf_report(raw), expected)

    def test_precise_decimals(self):
        raw = (
            "    99.99%    99.99%  ib_write_bw  libc.so.6  [.] write\n"
            "     0.01%     0.01%  ib_write_bw  libc.so.6  [.] read\n"
        )
        self.assertEqual(_parse_perf_report(raw), {"write@libc.so.6": 99.99, "read@libc.so.6": 0.01})

    def test_kernel_module_suffixes(self):
        raw = (
            "     7.22%     7.22%  ib_write_bw  [kernel.kallsyms]  [k] do_softirq.part.0\n"
            "     3.78%     3.78%  ib_write_bw  [kernel.kallsyms]  [k] finish_task_switch.isra.0\n"
        )
        self.assertEqual(_parse_perf_report(raw), {
            "do_softirq.part.0": 7.22,
            "finish_task_switch.isra.0": 3.78,
        })


# ---------------------------------------------------------------------------
# _parse_json_output  (5 scenarios)
# ---------------------------------------------------------------------------

class TestParseJsonOutput(unittest.TestCase):
    """_parse_json_output extracts the perftest JSON result."""

    def test_valid_dict(self):
        self.assertEqual(
            _parse_json_output('{"results": {"BW_average": 27.38}}'),
            {"results": {"BW_average": 27.38}},
        )

    def test_valid_list(self):
        self.assertEqual(
            _parse_json_output('[{"BW_average": 27.38}]'),
            {"result": [{"BW_average": 27.38}]},
        )

    def test_empty_string(self):
        result = _parse_json_output("")
        self.assertIn("error", result)
        self.assertIn("no JSON output", result["error"])

    def test_malformed_json(self):
        result = _parse_json_output("{bad json}")
        self.assertIn("error", result)
        self.assertIn("JSON parse failed", result["error"])

    def test_whitespace_only(self):
        result = _parse_json_output("   \n  ")
        self.assertIn("error", result)

    def test_bw_result_requires_bandwidth_metrics(self):
        result = {"results": {"BW_average": 27.38}}
        error = _validate_perftest_metrics("ib_write_bw", result)
        self.assertIn("MsgRate", error)

    def test_latency_result_requires_latency_metric(self):
        result = {"results": {"t_avg": 1.2}}
        self.assertEqual(_validate_perftest_metrics("ib_write_lat", result), "")

    def test_missing_results_object_is_metric_error(self):
        result = {"result": [{"BW_average": 27.38}]}
        self.assertIn("missing results object", _validate_perftest_metrics("ib_write_bw", result))


# ---------------------------------------------------------------------------
# Dual-host config and runner behavior
# ---------------------------------------------------------------------------

class TestDualHostConfig(unittest.TestCase):
    """Config normalization enforces a real client/server split."""

    def test_normalizes_preferred_schema(self):
        runtime = _runtime_config({
            "test": "ib_write_bw",
            "duration": 5,
            "server": {
                "host": "server-admin@example-server",
                "address": "rdma-server.example.com",
            },
            "client": {"host": "client-user@example-client"},
            "perftest": {
                "dir": "/opt/perftest",
                "rdma_core_lib": "/opt/rdma-core/build/lib",
                "env": {"FOO": "bar"},
                "json_file": "/tmp/custom.json",
            },
            "ssh": {
                "sudo": False,
                "connect_timeout": 3,
                "options": "-o StrictHostKeyChecking=no",
            },
            "report": {"title": "Two-node RDMA sweep"},
        })

        self.assertEqual(runtime["server"]["host"], "server-admin@example-server")
        self.assertEqual(runtime["server"]["address"], "rdma-server.example.com")
        self.assertEqual(runtime["client"]["host"], "client-user@example-client")
        self.assertEqual(runtime["perftest"]["dir"], "/opt/perftest")
        self.assertEqual(runtime["perftest"]["json_file"], "/tmp/custom.json")
        self.assertEqual(runtime["ssh"]["sudo"], False)
        self.assertEqual(runtime["ssh"]["options"], ["-o", "StrictHostKeyChecking=no"])
        self.assertEqual(runtime["report"]["title"], "Two-node RDMA sweep")

    def test_requires_client_host(self):
        with self.assertRaisesRegex(ValueError, "client.host"):
            _runtime_config({"server": {"host": "server-ssh", "address": "10.0.0.2"}})

    def test_requires_server_address(self):
        with self.assertRaisesRegex(ValueError, "server.address"):
            _runtime_config({
                "server": {"host": "server-ssh"},
                "client": {"host": "client-ssh"},
            })

    def test_parses_string_booleans(self):
        runtime = _runtime_config({
            "server": {"host": "server-ssh", "address": "10.0.0.2"},
            "client": {"host": "client-ssh"},
            "use_gpu": "false",
            "ssh": {"sudo": "false"},
            "perftest": {"dir": "/opt/perftest", "perf_record": "false"},
        })
        self.assertFalse(runtime["use_gpu"])
        self.assertFalse(runtime["ssh"]["sudo"])
        self.assertFalse(runtime["perftest"]["perf_record"])

    def test_rejects_ambiguous_string_boolean(self):
        with self.assertRaisesRegex(ValueError, "ssh.sudo"):
            _runtime_config({
                "server": {"host": "server-ssh", "address": "10.0.0.2"},
                "client": {"host": "client-ssh"},
                "perftest": {"dir": "/opt/perftest"},
                "ssh": {"sudo": "maybe"},
            })

    def test_requires_perftest_dir(self):
        with self.assertRaisesRegex(ValueError, "perftest.dir"):
            _runtime_config({
                "server": {"host": "server-ssh", "address": "10.0.0.2"},
                "client": {"host": "client-ssh"},
            })

    def test_ssh_defaults_verify_host_keys(self):
        runtime = _runtime_config({
            "server": {"host": "server-ssh", "address": "10.0.0.2"},
            "client": {"host": "client-ssh"},
            "perftest": {"dir": "/opt/perftest"},
        })
        self.assertFalse(runtime["ssh"]["allow_local"])
        self.assertIn("StrictHostKeyChecking=accept-new", runtime["ssh"]["options"])
        self.assertNotIn("StrictHostKeyChecking=no", runtime["ssh"]["options"])
        self.assertNotIn("UserKnownHostsFile=/dev/null", runtime["ssh"]["options"])

    def test_rejects_same_machine_even_with_different_users(self):
        with self.assertRaisesRegex(ValueError, "different machines"):
            _runtime_config({
                "server": {"host": "root@10.0.0.2"},
                "client": {"host": "user@10.0.0.2"},
            })

    def test_rejects_loopback_server_address(self):
        with self.assertRaisesRegex(ValueError, "not loopback"):
            _runtime_config({
                "server": {"host": "server-ssh", "address": "127.0.0.1"},
                "client": {"host": "client-ssh"},
            })


class TestBuildArgsConfigKeys(unittest.TestCase):
    """Runtime config keys are not forwarded as perftest flags."""

    def test_skips_runtime_config_keys_and_duration(self):
        args = _build_args({
            "msg_size": "64K",
            "qp": 4,
            "port": 18515,
            "duration": 5,
            "server": {"host": "server-ssh"},
            "client": {"host": "client-ssh"},
            "perftest": {"dir": "/tmp/perftest"},
            "ssh": {"sudo": False},
            "report": {"title": "x"},
            "server_address": "10.0.0.2",
            "device": "roceP2p1s0f1",
            "gid_index": 1,
            "force_link": "Ethernet",
            "rdma_cm": True,
            "comm_rdma_cm": False,
        })

        self.assertEqual(
            args,
            [
                "-s", "64K",
                "-q", "4",
                "-p", "18515",
                "-d", "roceP2p1s0f1",
                "-x", "1",
                "--force-link", "Ethernet",
                "-R",
            ],
        )
        self.assertNotIn("-D", args)
        self.assertNotIn("--server", args)
        self.assertNotIn("--client", args)


class TestRunPerftestDualHost(unittest.TestCase):
    """run_perftest starts server remotely and runs client against server.address."""

    def test_client_runs_on_client_host_against_server_address(self):
        perftest_config = dict(DEFAULT_PERFTEST_CONFIG)
        perftest_config.update({
            "dir": "/opt/perftest",
            "rdma_core_lib": "/opt/rdma-core/build/lib",
            "json_file": "/tmp/test_out.json",
            "time_file": "/tmp/test_time.out",
            "server_pid_file": "/tmp/test_server.pid",
            "server_log_file": "/tmp/test_server.log",
            "perf_data": "/tmp/test_perf.data",
            "perf_pid_file": "/tmp/test_perf.pid",
            "wait_timeout": 1,
        })
        calls = []

        def fake_run_remote_result(cmd, host, **kwargs):
            calls.append((host, cmd, kwargs))
            if "cat /tmp/test_server.pid" in cmd:
                return RemoteResult(host=host, command=cmd, stdout="123\n")
            if "cat /tmp/test_time.out" in cmd:
                return RemoteResult(host=host, command=cmd, stdout="1.0 0.5 50% 1024 0 0\n")
            if "perf report" in cmd:
                return RemoteResult(
                    host=host,
                    command=cmd,
                    stdout="10.00% 10.00% ib_write_bw [kernel.kallsyms] [k] rxe_requester\n",
                )
            if "cat /tmp/test_out.json" in cmd:
                return RemoteResult(
                    host=host,
                    command=cmd,
                    stdout='{"results": {"BW_average": 42, "MsgRate": 0.1}}',
                )
            return RemoteResult(host=host, command=cmd)

        with (
            patch("rdma_sweep._run_remote_result", side_effect=fake_run_remote_result),
            patch("rdma_sweep._wait_for_port") as wait_for_port,
            patch("rdma_sweep._cancel", return_value={}) as cancel,
        ):
            result = run_perftest(
                binary="ib_write_bw",
                server_host="server-ssh",
                client_host="client-ssh",
                server_address="10.0.0.2",
                perftest_config=perftest_config,
                ssh_config={"sudo": True, "connect_timeout": 1, "options": []},
                extra_args=["-s", "64K", "-q", "4", "-p", "18515"],
                duration=5,
                use_gpu=False,
            )

        self.assertEqual(result["results"]["BW_average"], 42)
        self.assertEqual(result["_process"]["server_host"], "server-ssh")
        self.assertEqual(result["_process"]["client_host"], "client-ssh")
        self.assertEqual(result["_process"]["server_address"], "10.0.0.2")
        wait_for_port.assert_called_once_with(
            "server-ssh",
            18515,
            timeout=1,
            ssh_config={"sudo": True, "connect_timeout": 1, "options": []},
        )
        cancel.assert_called_once()

        server_launches = [
            cmd for host, cmd, _ in calls
            if host == "server-ssh" and "/opt/perftest/ib_write_bw" in cmd and "/usr/bin/time" not in cmd
        ]
        client_runs = [
            cmd for host, cmd, _ in calls
            if host == "client-ssh" and "/usr/bin/time" in cmd
        ]
        json_reads = [
            cmd for host, cmd, _ in calls
            if host == "client-ssh" and "cat /tmp/test_out.json" in cmd
        ]

        self.assertEqual(len(server_launches), 1)
        self.assertEqual(len(client_runs), 1)
        self.assertEqual(len(json_reads), 1)
        self.assertNotIn("--server", server_launches[0])
        self.assertIn("10.0.0.2", client_runs[0])
        self.assertNotIn("127.0.0.1", client_runs[0])

    def test_iters_does_not_also_add_duration_flag(self):
        perftest_config = dict(DEFAULT_PERFTEST_CONFIG)
        perftest_config.update({
            "dir": "/opt/perftest",
            "json_file": "/tmp/test_out.json",
            "time_file": "/tmp/test_time.out",
            "server_pid_file": "/tmp/test_server.pid",
            "server_log_file": "/tmp/test_server.log",
            "perf_data": "/tmp/test_perf.data",
            "perf_pid_file": "/tmp/test_perf.pid",
            "perf_record": False,
            "wait_timeout": 1,
        })
        calls = []

        def fake_run_remote_result(cmd, host, **kwargs):
            calls.append((host, cmd, kwargs))
            if "cat /tmp/test_server.pid" in cmd:
                return RemoteResult(host=host, command=cmd, stdout="123\nnow\nrun\n")
            if "cat /tmp/test_time.out" in cmd:
                return RemoteResult(host=host, command=cmd, stdout="1.0 0.5 50% 1024 0 0\n")
            if "cat /tmp/test_out.json" in cmd:
                return RemoteResult(host=host, command=cmd, stdout='{"results": {"BW_average": 42, "MsgRate": 0.1}}')
            return RemoteResult(host=host, command=cmd)

        with (
            patch("rdma_sweep._run_remote_result", side_effect=fake_run_remote_result),
            patch("rdma_sweep._wait_for_port"),
            patch("rdma_sweep._cancel", return_value={}),
        ):
            result = run_perftest(
                binary="ib_write_bw",
                server_host="server-ssh",
                client_host="client-ssh",
                server_address="10.0.0.2",
                perftest_config=perftest_config,
                ssh_config={"sudo": False, "connect_timeout": 1, "options": []},
                extra_args=["-n", "100", "-p", "18515"],
                duration=5,
                use_gpu=False,
            )

        self.assertNotIn("error", result)
        relevant = [
            cmd for _host, cmd, _kwargs in calls
            if "/opt/perftest/ib_write_bw" in cmd
        ]
        self.assertTrue(relevant)
        self.assertTrue(all("-n 100" in cmd for cmd in relevant))
        self.assertTrue(all("-D 5" not in cmd for cmd in relevant))

    def test_client_nonzero_with_json_is_still_error(self):
        perftest_config = dict(DEFAULT_PERFTEST_CONFIG)
        perftest_config.update({
            "dir": "/opt/perftest",
            "json_file": "/tmp/test_out.json",
            "time_file": "/tmp/test_time.out",
            "server_pid_file": "/tmp/test_server.pid",
            "server_log_file": "/tmp/test_server.log",
            "perf_data": "/tmp/test_perf.data",
            "perf_pid_file": "/tmp/test_perf.pid",
            "perf_record": False,
            "wait_timeout": 1,
        })

        def fake_run_remote_result(cmd, host, **kwargs):
            if "cat /tmp/test_server.pid" in cmd:
                return RemoteResult(host=host, command=cmd, stdout="123\n")
            if "cat /tmp/test_time.out" in cmd:
                return RemoteResult(host=host, command=cmd, stdout="1.0 0.5 50% 1024 0 0\n")
            if "cat /tmp/test_out.json" in cmd:
                return RemoteResult(host=host, command=cmd, stdout='{"results": {"BW_average": 42, "MsgRate": 0.1}}')
            if "/usr/bin/time" in cmd:
                return RemoteResult(host=host, command=cmd, returncode=2, stderr="boom")
            return RemoteResult(host=host, command=cmd)

        with (
            patch("rdma_sweep._run_remote_result", side_effect=fake_run_remote_result),
            patch("rdma_sweep._wait_for_port"),
            patch("rdma_sweep._cancel", return_value={}),
        ):
            result = run_perftest(
                binary="ib_write_bw",
                server_host="server-ssh",
                client_host="client-ssh",
                server_address="10.0.0.2",
                perftest_config=perftest_config,
                ssh_config={"sudo": True, "connect_timeout": 1, "options": []},
                extra_args=["-s", "64K", "-q", "4", "-p", "18515"],
                duration=5,
                use_gpu=False,
            )

        self.assertIn("error", result)
        self.assertIn("client run failed", result["error"])

    def test_successful_client_missing_metric_is_still_error(self):
        perftest_config = dict(DEFAULT_PERFTEST_CONFIG)
        perftest_config.update({
            "dir": "/opt/perftest",
            "json_file": "/tmp/test_out.json",
            "time_file": "/tmp/test_time.out",
            "server_pid_file": "/tmp/test_server.pid",
            "server_log_file": "/tmp/test_server.log",
            "perf_data": "/tmp/test_perf.data",
            "perf_pid_file": "/tmp/test_perf.pid",
            "perf_record": False,
            "wait_timeout": 1,
        })

        def fake_run_remote_result(cmd: str, host: str, **kwargs: object) -> RemoteResult:
            if "cat /tmp/test_server.pid" in cmd:
                return RemoteResult(host=host, command=cmd, stdout="123\n")
            if "cat /tmp/test_time.out" in cmd:
                return RemoteResult(host=host, command=cmd, stdout="1.0 0.5 50% 1024 0 0\n")
            if "cat /tmp/test_out.json" in cmd:
                return RemoteResult(host=host, command=cmd, stdout='{"results": {"MsgRate": 0.1}}')
            return RemoteResult(host=host, command=cmd)

        with (
            patch("rdma_sweep._run_remote_result", side_effect=fake_run_remote_result),
            patch("rdma_sweep._wait_for_port"),
            patch("rdma_sweep._cancel", return_value={}),
        ):
            result = run_perftest(
                binary="ib_write_bw",
                server_host="server-ssh",
                client_host="client-ssh",
                server_address="10.0.0.2",
                perftest_config=perftest_config,
                ssh_config={"sudo": True, "connect_timeout": 1, "options": []},
                extra_args=["-s", "64K", "-q", "4", "-p", "18515"],
                duration=5,
                use_gpu=False,
            )

        self.assertIn("error", result)
        self.assertIn("BW_average", result["error"])


class TestRemoteCommandResults(unittest.TestCase):
    """Remote command evidence stays serializable and dual-host by default."""

    def test_timeout_output_bytes_are_json_serializable(self):
        timeout = subprocess.TimeoutExpired(
            cmd=["bash", "-c", "sleep 2"],
            timeout=1,
            output=b"partial stdout",
            stderr=b"partial stderr",
        )
        with patch("rdma_remote.subprocess.run", side_effect=timeout):
            result = run_local_result("sleep 2", timeout=1, sudo=False)

        payload = result.to_dict()
        json.dumps(payload)
        self.assertEqual(payload["stdout"], "partial stdout")
        self.assertEqual(payload["stderr"], "partial stderr")
        self.assertTrue(payload["timed_out"])

    def test_local_host_uses_ssh_unless_allow_local_is_explicit(self):
        completed = subprocess.CompletedProcess(
            args=["ssh"],
            returncode=0,
            stdout="ok",
            stderr="",
        )
        with (
            patch("rdma_remote._LOCAL_HOSTS", {"localhost"}),
            patch("rdma_remote.subprocess.run", return_value=completed) as run,
        ):
            result = run_remote_result(
                "echo ok",
                "localhost",
                ssh_config={"sudo": False, "connect_timeout": 1, "options": []},
            )

        self.assertTrue(result.ok)
        self.assertEqual(run.call_args.args[0][0], "ssh")

    def test_allow_local_uses_local_runner(self):
        completed = subprocess.CompletedProcess(
            args=["bash", "-c", "echo ok"],
            returncode=0,
            stdout="ok\n",
            stderr="",
        )
        with (
            patch("rdma_remote._LOCAL_HOSTS", {"localhost"}),
            patch("rdma_remote.subprocess.run", return_value=completed) as run,
        ):
            result = run_remote_result(
                "echo ok",
                "localhost",
                ssh_config={"sudo": False, "allow_local": True},
            )

        self.assertTrue(result.ok)
        self.assertEqual(run.call_args.args[0][0], "bash")

    def test_cancel_returns_cleanup_failure_evidence(self):
        def fake_run_remote_result(cmd, host, **kwargs):
            if cmd.startswith("cat "):
                return RemoteResult(
                    host=host,
                    command=cmd,
                    stdout="123\nMon Jun 22 10:00:00 2026\nrun-1\n",
                )
            return RemoteResult(
                host=host,
                command=cmd,
                returncode=1,
                stderr="cleanup failed for pid 123",
            )

        with patch("rdma_sweep._run_remote_result", side_effect=fake_run_remote_result):
            evidence = _cancel(
                "server-ssh",
                {"sudo": True},
                "/tmp/server.pid",
                "/opt/perftest/ib_write_bw",
                "run-1",
            )

        self.assertIn("pid_read", evidence)
        self.assertIn("cleanup", evidence)
        self.assertIn("cleanup failed", evidence["error"])


class TestReportGeneration(unittest.TestCase):
    """Report generation fails clearly for invalid summaries."""

    def test_empty_summary_fails_clearly(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "summary.json").write_text("[]")
            with self.assertRaisesRegex(ValueError, "no sweep entries"):
                generate_report(str(out))


# ---------------------------------------------------------------------------
# SysMonitor  (7 scenarios)
# ---------------------------------------------------------------------------

class TestSysMonitorSoftirqDiff(unittest.TestCase):
    """SysMonitor.compute_softirq_diff delta computation."""

    def test_basic_delta(self):
        b = {"softirqs": {"NET_RX": 1000, "NET_TX": 500, "TASKLET": 200}}
        a = {"softirqs": {"NET_RX": 1050, "NET_TX": 530, "TASKLET": 210}}
        self.assertEqual(SysMonitor.compute_softirq_diff(b, a),
                         {"NET_RX": 50, "NET_TX": 30, "TASKLET": 10})

    def test_empty_before_returns_empty(self):
        self.assertEqual(SysMonitor.compute_softirq_diff({},
                         {"softirqs": {"NET_RX": 1050}}), {})

    def test_after_missing_key_excluded(self):
        b = {"softirqs": {"NET_RX": 1000}}
        a = {"softirqs": {"NET_RX": 1050, "NET_TX": 500}}
        self.assertEqual(SysMonitor.compute_softirq_diff(b, a),
                         {"NET_RX": 50})

    def test_identical_before_after(self):
        b = {"softirqs": {"NET_RX": 1000}}
        self.assertEqual(SysMonitor.compute_softirq_diff(b, b),
                         {"NET_RX": 0})

    def test_multiple_softirq_types(self):
        b = {"softirqs": {"HI": 10, "TIMER": 500, "NET_TX": 300, "NET_RX": 1000,
                          "BLOCK": 50, "IRQ_POLL": 0, "TASKLET": 200, "SCHED": 800,
                          "HRTIMER": 100, "RCU": 3000}}
        a = {"softirqs": {"HI": 10, "TIMER": 510, "NET_TX": 330, "NET_RX": 1050,
                          "BLOCK": 55, "IRQ_POLL": 0, "TASKLET": 210, "SCHED": 820,
                          "HRTIMER": 100, "RCU": 3100}}
        expected = {"HI": 0, "TIMER": 10, "NET_TX": 30, "NET_RX": 50,
                    "BLOCK": 5, "IRQ_POLL": 0, "TASKLET": 10, "SCHED": 20,
                    "HRTIMER": 0, "RCU": 100}
        self.assertEqual(SysMonitor.compute_softirq_diff(b, a), expected)


class TestSysMonitorCpuDiff(unittest.TestCase):
    """SysMonitor.compute_cpu_diff reads ``cores`` key (SysMonitor.grab() format)."""

    def test_basic_delta(self):
        b = {"cores": {"cpu0": {"user": 1000, "nice": 0, "system": 500, "idle": 50000, "iowait": 0}}}
        a = {"cores": {"cpu0": {"user": 1050, "nice": 0, "system": 530, "idle": 50010, "iowait": 0}}}
        r = SysMonitor.compute_cpu_diff(b, a)
        # total delta = 50+0+30+10+0 = 90.  idle delta = 10.  util = 100*(1 - 10/90) ≈ 88.89
        self.assertIn("cpu0", r)
        self.assertGreater(r["cpu0"], 85.0)
        self.assertLess(r["cpu0"], 95.0)

    def test_no_delta_idle(self):
        b = {"cores": {"cpu0": {"user": 1000, "nice": 0, "system": 500, "idle": 50000, "iowait": 0}}}
        a = {"cores": {"cpu0": {"user": 1000, "nice": 0, "system": 500, "idle": 50000, "iowait": 0}}}
        r = SysMonitor.compute_cpu_diff(b, a)
        self.assertEqual(r["cpu0"], 0.0)


    def test_demangled_cpp_symbol_with_spaces(self):
        """Codex found: parts[-1] loses demangled names like 'operator new(unsigned long)'"""
        raw = "    10.00%    10.00%  ib_write_bw  libstdc++.so  [.] operator new(unsigned long)\n"
        result = _parse_perf_report(raw)
        self.assertIn("operator new(unsigned long)@libstdc++.so", result)
        self.assertEqual(result["operator new(unsigned long)@libstdc++.so"], 10.0)

    def test_duplicate_symbol_different_dso(self):
        """Codex found: poll from liba.so vs libb.so silently overwritten"""
        raw = (
            "     8.00%     8.00%  ib_write_bw  liba.so  [.] poll\n"
            "     7.00%     7.00%  ib_write_bw  libb.so  [.] poll\n"
        )
        result = _parse_perf_report(raw)
        self.assertEqual(result["poll@liba.so"], 8.0)
        self.assertEqual(result["poll@libb.so"], 7.0)


if __name__ == "__main__":
    unittest.main()
