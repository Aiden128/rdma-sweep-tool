"""Tests for rdma_sweep — unit-testable components.

Run from this directory: python3 -m unittest test_rdma_sweep
(A single case: python3 -m unittest test_rdma_sweep.TestSummaryAttribution)
"""

import io
import shutil
import unittest
import sys
import os
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(__file__))

from rdma_config import DEFAULT_PERFTEST_CONFIG
from rdma_sweep import (
    SysMonitor,
    _build_args,
    _build_perftest_cmdline,
    _cancel,
    _chart_area,
    _chart_column_pos,
    _chart_content_box,
    _chart_empty_state,
    _chart_legend_item,
    _chart_panel,
    _chart_panel_rect,
    _chart_sys_series,
    _chart_x_label,
    _chart_y_label,
    _compute_node_deltas,
    _config_int,
    _core_cell,
    _record_command,
    _run_cleanup_cmd,
    _dry_run_print,
    _draw_series_polyline,
    _draw_y_grid,
    _enrich_error_with_server_log,
    _expand,
    _extract_metric,
    _filter_positive_x,
    _get_dict,
    _handle_run_exception,
    _launch_server_and_get_pid,
    _load_perf_bar_series,
    _make_process_tracker,
    _make_server_error_result,
    _mem_delta_mib,
    _mem_used_kb,
    _or_err,
    _read_client_time_usage,
    _read_json,
    _parse_int_fields,
    _parse_json_output,
    _parse_pid_file_lines,
    _parse_perf_line,
    _parse_perf_report,
    _parse_proc_meminfo,
    _parse_proc_softirqs,
    _parse_proc_stat,
    _port_from_args,
    _run_one_combo,
    _runtime_config,
    _scale_y,
    _summary_entry,
    _svg_chart,
    _svg_empty_panel,
    _svg_placeholder_text,
    _svg_title,
    _sys_ok,
    _validate_perftest_metrics,
    _wait_for_port,
    _write_csv,
    _write_json,
    generate_report,
    main,
    run_perftest,
)
from rdma_remote import RemoteResult, run_local_result, run_remote_result


def _remote_fake(**responses):
    """Factory for simple substring-dispatch fakes that don't record calls.

    Each key in *responses* is a substring matched against the command string.
    The value may be a RemoteResult (returned as-is) or a callable
    ``(cmd, host) -> RemoteResult``.  Unmatched commands return an ok
    RemoteResult with empty stdout (the default run_remote_result success).
    """
    def fake(cmd, host, timeout=300, ssh_config=None, sudo=None):
        for key, val in responses.items():
            if key in cmd:
                return val(cmd, host) if callable(val) else val
        return RemoteResult(host=host, command=cmd)
    return fake


# ---------------------------------------------------------------------------
# _parse_int_fields  (5 scenarios)
# ---------------------------------------------------------------------------

class TestParseIntFields(unittest.TestCase):
    """_parse_int_fields converts parts[1:] to ints, returning None on failure."""

    def test_returns_ints(self):
        """Valid numeric strings return list of ints."""
        result = _parse_int_fields(["cpu", "1", "2", "3"])
        self.assertEqual(result, [1, 2, 3])

    def test_returns_none_on_non_numeric(self):
        """Non-numeric field causes None return."""
        result = _parse_int_fields(["cpu", "a", "2"])
        self.assertIsNone(result)

    def test_empty_suffix(self):
        """No fields after parts[0] returns empty list."""
        result = _parse_int_fields(["cpu"])
        self.assertEqual(result, [])

    def test_mixed_valid_invalid(self):
        """Mixed valid/invalid returns None (all-or-nothing)."""
        result = _parse_int_fields(["cpu", "1", "bad", "3"])
        self.assertIsNone(result)

    def test_negative_ints(self):
        """Negative integers are valid."""
        result = _parse_int_fields(["cpu", "-1", "5"])
        self.assertEqual(result, [-1, 5])

    def test_empty_token_returns_none(self):
        """Empty string token (e.g. split artifact) raises ValueError → None."""
        result = _parse_int_fields(["cpu", ""])
        self.assertIsNone(result)

    def test_float_token_returns_none(self):
        """Float-string '5.0' fails int() → None."""
        result = _parse_int_fields(["cpu", "5.0"])
        self.assertIsNone(result)


class TestMemUsedKb(unittest.TestCase):
    """_mem_used_kb computes used=MemTotal-MemFree-Buffers-Cached."""

    def test_normal_case(self):
        """Standard meminfo dict returns correct used kB."""
        m = {"MemTotal": 10000, "MemFree": 2000, "Buffers": 500, "Cached": 1500}
        self.assertEqual(_mem_used_kb(m), 6000)

    def test_missing_keys_default_to_zero(self):
        """Missing keys are treated as 0 (graceful degradation)."""
        self.assertEqual(_mem_used_kb({"MemTotal": 5000}), 5000)

    def test_empty_dict(self):
        """Empty dict returns 0 (all keys default to 0)."""
        self.assertEqual(_mem_used_kb({}), 0)

    def test_negative_if_cache_exceeds_total(self):
        """Can return negative if cache > total (callers handle this)."""
        m = {"MemTotal": 1000, "MemFree": 100, "Buffers": 0, "Cached": 2000}
        self.assertEqual(_mem_used_kb(m), -1100)


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

    def test_skips_callgraph_child_lines(self):
        raw = (
            "    35.20%    35.20%  ib_write_bw  ib_write_bw  [.] run_iterations\n"
            "            |\n"
            "            ---0\n"
            "               ibv_post_send\n"
            "               ibv_poll_cq\n"
        )
        self.assertEqual(_parse_perf_report(raw), {"run_iterations@ib_write_bw": 35.2})

    def test_embedded_marker_token_skipped_not_crash(self):
        # A demangled symbol can embed "[.]"/"[k]" INSIDE a token (no
        # standalone marker), which passes the substring line-guard.  A bare
        # next() would then raise StopIteration -> caught upstream -> the whole
        # run's valid bandwidth + perf profile discarded.  Such a line must be
        # skipped, and well-formed lines still parsed.
        raw = (
            "    35.20%    35.20%  ib_write_bw  ib_write_bw  [.] run_iterations\n"
            "    10.00%    10.00%  ib_write_bw  some.so  weird[.]embedded extra\n"
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

    def test_skips_line_with_unparseable_self_pct(self):
        """A line with [k] marker but non-numeric self% is skipped (ValueError)."""
        raw = (
            "    35.20%    35.20%  ib_write_bw  ib_write_bw  [.] run_iterations\n"
            "     8.50%  INVALID%  ib_write_bw  [kernel.kallsyms]  [k] rxe_requester\n"
            "     3.90%     3.90%  ib_write_bw  [kernel.kallsyms]  [k] rxe_send\n"
        )
        self.assertEqual(_parse_perf_report(raw), {
            "run_iterations@ib_write_bw": 35.2,
            "rxe_send": 3.9,
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

    def test_rejects_valueless_config_key(self):
        # A YAML key written with no value parses to None (e.g. `inline:`) or
        # an empty string (`inline: ""`).  Forwarding it would emit
        # ["-I", "None"]/["-I", ""] and surface a cryptic perftest failure far
        # from the typo.  _build_args must fail loud at the source instead.
        for bad in (None, "", "   ", [], {}):
            with self.assertRaises(ValueError) as ctx:
                _build_args({"inline": bad})
            self.assertIn("inline", str(ctx.exception))
        # The error names the offending key, not a fixed one.
        with self.assertRaises(ValueError) as ctx:
            _build_args({"cq_mod": None})
        self.assertIn("cq_mod", str(ctx.exception))

    def test_valueless_toggle_and_zero_not_rejected(self):
        # The guard must catch ONLY None/empty -- never an explicit boolean
        # toggle (the legitimate no-value flag) nor a real 0 value.
        self.assertEqual(_build_args({"rdma_cm": True}), ["-R"])
        self.assertEqual(_build_args({"comm_rdma_cm": False}), [])
        self.assertEqual(_build_args({"iters": 0}), ["-n", "0"])
        # A NON-empty container holding only falsy elements is still a value:
        # only genuinely-empty []/{} are "no value".  Pins the boundary so a
        # future "simplify to bare `not v`" refactor cannot silently regress.
        self.assertEqual(_build_args({"inline": [0]}), ["-I", "[0]"])


class TestDeepMergeDoesNotAliasGlobals(unittest.TestCase):
    """deep_merge must return fresh nested objects, not aliases to module-level defaults.

    The old implementation used ``dict(base)`` (shallow copy), which left nested
    mutables (like DEFAULT_PERFTEST_CONFIG["env"] = {} or
    DEFAULT_SSH_CONFIG["options"] = [...]) pointing at the module global.
    Mutating the returned config silently corrupted the default for every
    subsequent call — a cross-run pollution that was nigh impossible to debug.
    """

    def test_env_is_not_the_global(self):
        from rdma_config import deep_merge, DEFAULT_PERFTEST_CONFIG
        merged = deep_merge(DEFAULT_PERFTEST_CONFIG, {})
        merged["env"]["FOO"] = "bar"
        self.assertNotIn("FOO", DEFAULT_PERFTEST_CONFIG["env"])

    def test_options_list_is_not_the_global(self):
        from rdma_config import deep_merge, DEFAULT_SSH_CONFIG
        merged = deep_merge(DEFAULT_SSH_CONFIG, {})
        merged["options"].append("-o Fake=yes")
        self.assertNotIn("-o Fake=yes", DEFAULT_SSH_CONFIG["options"])


class TestRemoteResult(unittest.TestCase):
    """RemoteResult helpers contract."""

    def test_error_summary_returns_empty_on_success(self):
        r = RemoteResult(host="x", command="true", returncode=0)
        self.assertTrue(r.ok)
        self.assertEqual(r.error_summary(), "")

    def test_error_summary_returns_stderr_only_on_failure(self):
        r = RemoteResult(host="x", command="false", returncode=1, stderr="oops")
        self.assertFalse(r.ok)
        self.assertEqual(r.error_summary(), "oops")

    def test_error_summary_ignores_stderr_on_success(self):
        # Command exited 0 but printed to stderr (e.g., a warning).
        # error_summary must not return that stderr as an "error".
        r = RemoteResult(host="x", command="warn", returncode=0, stderr="warning here")
        self.assertTrue(r.ok)
        self.assertEqual(r.error_summary(), "")


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

        # Signature-faithful fake (no **kwargs): a leftover/renamed kwarg in
        # run_perftest's body raises TypeError here even if autospec= is ever
        # dropped from the patch() below. See TestWaitForPort docstring.
        def fake_run_remote_result(cmd, host, timeout=300, ssh_config=None, sudo=None):
            calls.append((host, cmd, {"timeout": timeout, "ssh_config": ssh_config, "sudo": sudo}))
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
            patch("rdma_sweep._run_remote_result", autospec=True, side_effect=fake_run_remote_result),
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

        # Signature-faithful fake (no **kwargs): see TestWaitForPort docstring.
        def fake_run_remote_result(cmd, host, timeout=300, ssh_config=None, sudo=None):
            calls.append((host, cmd, {"timeout": timeout, "ssh_config": ssh_config, "sudo": sudo}))
            if "cat /tmp/test_server.pid" in cmd:
                return RemoteResult(host=host, command=cmd, stdout="123\nnow\nrun\n")
            if "cat /tmp/test_time.out" in cmd:
                return RemoteResult(host=host, command=cmd, stdout="1.0 0.5 50% 1024 0 0\n")
            if "cat /tmp/test_out.json" in cmd:
                return RemoteResult(host=host, command=cmd, stdout='{"results": {"BW_average": 42, "MsgRate": 0.1}}')
            return RemoteResult(host=host, command=cmd)

        with (
            patch("rdma_sweep._run_remote_result", autospec=True, side_effect=fake_run_remote_result),
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

        fake_run_remote_result = _remote_fake(**{
            "cat /tmp/test_server.pid": RemoteResult(host="", command="", stdout="123\n"),
            "cat /tmp/test_time.out":   RemoteResult(host="", command="", stdout="1.0 0.5 50% 1024 0 0\n"),
            "cat /tmp/test_out.json":   RemoteResult(host="", command="", stdout='{"results": {"BW_average": 42, "MsgRate": 0.1}}'),
            "/usr/bin/time":            RemoteResult(host="", command="", returncode=2, stderr="boom"),
        })

        with (
            patch("rdma_sweep._run_remote_result", autospec=True, side_effect=fake_run_remote_result),
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

    def test_client_json_read_failure_is_detected(self):
        """When the JSON result file is missing, cat exits non-zero → json_read.ok is
        False → the error is attributed to 'client JSON read failed' (not falling
        through to a misleading 'perftest JSON missing results object')."""
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

        fake_run_remote_result = _remote_fake(**{
            "cat /tmp/test_server.pid": RemoteResult(host="", command="", stdout="123\n"),
            "cat /tmp/test_time.out":   RemoteResult(host="", command="", stdout="1.0 0.5 50% 1024 0 0\n"),
            # JSON file is missing — cat exits 1 with empty stdout
            "cat /tmp/test_out.json":   RemoteResult(host="", command="", returncode=1),
            "/usr/bin/time":            RemoteResult(host="", command="", stdout="1.0 0.5 50% 1024 0 0\n"),
        })

        with (
            patch("rdma_sweep._run_remote_result", autospec=True, side_effect=fake_run_remote_result),
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
        self.assertIn("client JSON read failed", result["error"])

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

        fake_run_remote_result = _remote_fake(**{
            "cat /tmp/test_server.pid": RemoteResult(host="", command="", stdout="123\n"),
            "cat /tmp/test_time.out":   RemoteResult(host="", command="", stdout="1.0 0.5 50% 1024 0 0\n"),
            "cat /tmp/test_out.json":   RemoteResult(host="", command="", stdout='{"results": {"MsgRate": 0.1}}'),
        })

        with (
            patch("rdma_sweep._run_remote_result", autospec=True, side_effect=fake_run_remote_result),
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

    def test_server_start_exception_stores_log_tail_error(self):
        """Exception path (lines 671-673): server_start raises, then _fetch_server_log_tail
        inside the except handler also raises — process["server_log_tail_error"] is set.

        This tests the INNER exception handler inside the outer except block.
        """
        perftest_config = dict(DEFAULT_PERFTEST_CONFIG)
        perftest_config.update({
            "dir": "/opt/perftest",
            "json_file": "/tmp/test_out.json",
            "time_file": "/tmp/test_time.out",
            "server_pid_file": "/tmp/test_server.pid",
            "server_log_file": "/tmp/test_server.log",
            "perf_data": "/tmp/test_perf.data",
            "perf_pid_file": "/tmp/test_perf.pid",
            "wait_timeout": 1,
        })

        def fake_raise(*args, **kwargs):
            raise RuntimeError("SSH connection lost")

        with patch("rdma_sweep._run_remote_result", autospec=True, side_effect=fake_raise):
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
        self.assertIn("SSH connection lost", result["error"])
        self.assertIn("_process", result)
        # Outer exception handler stored the error
        self.assertIn("exception", result["_process"])
        # Inner exception handler (lines 671-673) stored _fetch_server_log_tail's error
        self.assertIn("server_log_tail_error", result["_process"])
        # _cancel exception handling stored perf_error
        self.assertIn("cleanup", result["_process"].get("commands", {}))
        self.assertIn("perf_error", result["_process"]["commands"].get("cleanup", {}))

    def test_server_start_exception_with_successful_log_tail(self):
        """Exception path (line 671): server_start raises, but _fetch_server_log_tail
        inside the except handler succeeds — process["server_log_tail"] is set.

        This tests the SUCCESS path of _fetch_server_log_tail inside the outer except.
        """
        perftest_config = dict(DEFAULT_PERFTEST_CONFIG)
        perftest_config.update({
            "dir": "/opt/perftest",
            "json_file": "/tmp/test_out.json",
            "time_file": "/tmp/test_time.out",
            "server_pid_file": "/tmp/test_server.pid",
            "server_log_file": "/tmp/test_server.log",
            "perf_data": "/tmp/test_perf.data",
            "perf_pid_file": "/tmp/test_perf.pid",
            "wait_timeout": 1,
        })

        def selective_fake(cmd, host, timeout=300, ssh_config=None, sudo=None):
            # server_start begins with mkdir — raise to trigger the outer except
            if cmd.startswith("mkdir -p"):
                raise RuntimeError("SSH connection lost during server start")
            # _fetch_server_log_tail uses tail — return log content
            if cmd.startswith("tail -"):
                return RemoteResult(host=host, command=cmd, stdout="server log content\n")
            # Everything else (_cancel etc.) — return clean result
            return RemoteResult(host=host, command=cmd)

        with patch("rdma_sweep._run_remote_result", autospec=True, side_effect=selective_fake):
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
        self.assertIn("SSH connection lost", result["error"])
        self.assertIn("_process", result)
        # _fetch_server_log_tail succeeded — line 671 was hit
        self.assertIn("server_log_tail", result["_process"])
        self.assertEqual(result["_process"]["server_log_tail"], "server log content")

    def test_server_start_non_zero_exit_returns_error_early(self):
        """server_start returns non-ok RemoteResult (line 547): early return with
        'server launch failed' error — not an exception path, just a non-zero SSH exit.

        This is distinct from a _run_remote_result *raise* (tested in
        test_server_start_exception_*): the SSH command completed but the remote
        host reported failure (e.g., binary not found).
        """
        perftest_config = dict(DEFAULT_PERFTEST_CONFIG)
        perftest_config.update({
            "dir": "/opt/perftest",
            "json_file": "/tmp/test_out.json",
            "time_file": "/tmp/test_time.out",
            "server_pid_file": "/tmp/test_server.pid",
            "server_log_file": "/tmp/test_server.log",
            "perf_data": "/tmp/test_perf.data",
            "perf_pid_file": "/tmp/test_perf.pid",
            "wait_timeout": 1,
        })

        def fake_run(cmd, host, timeout=300, ssh_config=None, sudo=None):
            # server_start command fails (binary not found on remote host)
            if cmd.startswith("mkdir -p"):
                return RemoteResult(
                    host=host, command=cmd, returncode=1,
                    stderr="bash: ib_write_bw: command not found",
                )
            return RemoteResult(host=host, command=cmd)

        with (
            patch("rdma_sweep._run_remote_result", autospec=True, side_effect=fake_run),
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
        self.assertEqual(result["error"], "server launch failed: bash: ib_write_bw: command not found")
        # server_start command is logged before the early return
        self.assertIn("server_start", result.get("_process", {}).get("commands", {}))
        # pid_read should NOT be present (we returned before that code)
        self.assertNotIn("server_pid_read", result.get("_process", {}).get("commands", {}))


class TestRunPerftestPerfRecord(unittest.TestCase):
    """run_perftest(perf_record=True) drives the perf record/stop/report path.

    No other test sets perf_record=True, so this whole branch — the
    ``perf record -g`` launch, the SIGINT stop, the ``perf report`` read, and
    the parse-into result["_process"]["server_perf"] — never executed in the
    suite.  That is the exact 'mocked-away body' class that let the historical
    ``check=`` regression ship green.  The fakes are signature-faithful (no
    ``**kwargs``) so a leftover/renamed kwarg on ANY perf-path
    _run_remote_result call raises TypeError here, not only at runtime.
    """

    def _perf_config(self):
        cfg = dict(DEFAULT_PERFTEST_CONFIG)
        cfg.update({
            "dir": "/opt/perftest",
            "json_file": "/tmp/test_out.json",
            "time_file": "/tmp/test_time.out",
            "server_pid_file": "/tmp/test_server.pid",
            "server_log_file": "/tmp/test_server.log",
            "perf_data": "/tmp/test_perf.data",
            "perf_pid_file": "/tmp/test_perf.pid",
            "perf_record": True,
            "wait_timeout": 1,
        })
        return cfg

    def _run(self, fake, use_gpu=False, perf_config=None):
        if perf_config is None:
            perf_config = self._perf_config()
        with (
            patch("rdma_sweep._run_remote_result", autospec=True, side_effect=fake),
            patch("rdma_sweep._wait_for_port"),
            patch("rdma_sweep._cancel", return_value={}),
        ):
            return run_perftest(
                binary="ib_write_bw",
                server_host="server-ssh",
                client_host="client-ssh",
                server_address="10.0.0.2",
                perftest_config=perf_config,
                ssh_config={"sudo": True, "connect_timeout": 1, "options": []},
                extra_args=["-s", "64K", "-q", "4", "-p", "18515"],
                duration=5,
                use_gpu=use_gpu,
            )

    def test_perf_record_happy_path_parses_and_stores_symbols(self):
        calls = []

        def fake(cmd, host, timeout=300, ssh_config=None, sudo=None):
            calls.append(cmd)
            if "cat /tmp/test_server.pid" in cmd:
                return RemoteResult(host=host, command=cmd,
                                    stdout="123\nMon Jun 22 10:00:00 2026\nrun-1\n")
            if "cat /tmp/test_time.out" in cmd:
                return RemoteResult(host=host, command=cmd, stdout="1.0 0.5 50% 1024 0 0\n")
            if "perf report --stdio" in cmd:
                return RemoteResult(
                    host=host, command=cmd,
                    stdout="    8.50%     8.50%  ib_write_bw  [kernel.kallsyms]  [k] rxe_requester\n",
                )
            if "cat /tmp/test_out.json" in cmd:
                return RemoteResult(host=host, command=cmd,
                                    stdout='{"results": {"BW_average": 42, "MsgRate": 0.1}}')
            return RemoteResult(host=host, command=cmd)

        result = self._run(fake)

        # The primary measurement still succeeds ...
        self.assertNotIn("error", result)
        self.assertEqual(result["results"]["BW_average"], 42)
        # ... AND the perf orchestration ran end-to-end: a -g record on the
        # server PID, a SIGINT stop, a report read, and the parsed symbol stored.
        self.assertEqual(result["_process"]["server_perf"], {"rxe_requester": 8.5})
        self.assertTrue(any("perf record -g -p 123" in c for c in calls))
        self.assertTrue(any("kill -INT" in c for c in calls))
        self.assertTrue(any("perf report --stdio --no-header -g none" in c for c in calls))

    def test_perf_record_start_failure_does_not_abort_measurement(self):
        # ``perf record`` is diagnostic overhead — if it fails (e.g. missing
        # kernel support, permissions), the bandwidth measurement must still
        # proceed and the perf failure is recorded in process metadata.
        def fake(cmd, host, timeout=300, ssh_config=None, sudo=None):
            if "cat /tmp/test_server.pid" in cmd:
                return RemoteResult(host=host, command=cmd, stdout="123\n")
            if "perf record -g" in cmd:
                return RemoteResult(host=host, command=cmd, returncode=1,
                                    stderr="perf: command not found")
            if "cat /tmp/test_time.out" in cmd:
                return RemoteResult(host=host, command=cmd, stdout="1.0 0.5 50% 1024 0 0\n")
            if "cat /tmp/test_out.json" in cmd:
                return RemoteResult(host=host, command=cmd,
                                    stdout='{"results": {"BW_average": 42, "MsgRate": 0.1}}')
            return RemoteResult(host=host, command=cmd)

        result = self._run(fake)
        # Bandwidth measurement still succeeds ...
        self.assertNotIn("error", result)
        self.assertEqual(result["results"]["BW_average"], 42)
        # ... and the perf failure is logged in process metadata.
        self.assertIn("perf_start_error", result["_process"])

    def test_perf_report_failure_preserves_bw_and_flags_error(self):
        # A failed ``perf report`` (e.g. corrupt/missing perf.data) must NOT
        # discard an already-valid BW measurement.  Mirror the perf-start path:
        # preserve BW and record the diagnostic failure in _process.
        def fake(cmd, host, timeout=300, ssh_config=None, sudo=None):
            if "cat /tmp/test_server.pid" in cmd:
                return RemoteResult(host=host, command=cmd, stdout="123\n")
            if "cat /tmp/test_time.out" in cmd:
                return RemoteResult(host=host, command=cmd, stdout="1.0 0.5 50% 1024 0 0\n")
            if "perf report --stdio" in cmd:
                return RemoteResult(host=host, command=cmd, returncode=1,
                                    stderr="failed to open perf.data")
            if "cat /tmp/test_out.json" in cmd:
                return RemoteResult(host=host, command=cmd,
                                    stdout='{"results": {"BW_average": 42, "MsgRate": 0.1}}')
            return RemoteResult(host=host, command=cmd)

        result = self._run(fake)
        # BW preserved — valid measurement not discarded
        self.assertNotIn("error", result)
        self.assertEqual(result["results"]["BW_average"], 42)
        # perf failure attributed in _process, not silently swallowed
        self.assertIn("perf_collection_error", result["_process"])
        self.assertIn("perf report failed", result["_process"]["perf_collection_error"])

    def test_perf_stop_failure_preserves_bw_and_flags_error(self):
        # A failed SIGINT-stop must NOT discard an already-valid BW measurement.
        # Mirror the perf-start path: preserve BW and record the diagnostic failure.
        def fake(cmd, host, timeout=300, ssh_config=None, sudo=None):
            if "cat /tmp/test_server.pid" in cmd:
                return RemoteResult(host=host, command=cmd, stdout="123\n")
            if "cat /tmp/test_time.out" in cmd:
                return RemoteResult(host=host, command=cmd, stdout="1.0 0.5 50% 1024 0 0\n")
            if "kill -INT" in cmd:  # the perf_stop command
                return RemoteResult(host=host, command=cmd, returncode=1,
                                    stderr="kill: no such process")
            if "cat /tmp/test_out.json" in cmd:
                return RemoteResult(host=host, command=cmd,
                                    stdout='{"results": {"BW_average": 42, "MsgRate": 0.1}}')
            return RemoteResult(host=host, command=cmd)

        result = self._run(fake)
        # BW preserved — valid measurement not discarded
        self.assertNotIn("error", result)
        self.assertEqual(result["results"]["BW_average"], 42)
        # perf failure attributed in _process
        self.assertIn("perf_collection_error", result["_process"])
        self.assertIn("perf stop failed", result["_process"]["perf_collection_error"])

    def test_perf_unreadable_pid_attributed_not_silently_skipped(self):
        # perf_record=True but the server PID is empty/non-numeric while the port
        # still came up: the run is NOT aborted (BW is valid) and no ``perf
        # record`` is issued -- but the miss must be ATTRIBUTED as
        # ``perf_start_error``, not silently dropped.  Otherwise the empty
        # server_perf renders as a zero-height "no CPU consumers" bar (a
        # masquerade) instead of "n/a".  The error key is what flips the bar to
        # "n/a" downstream (see test_svg_failed_profile_*); without this
        # attribution the run is indistinguishable from perf_record=False.
        calls = []

        def fake(cmd, host, timeout=300, ssh_config=None, sudo=None):
            calls.append(cmd)
            if "cat /tmp/test_server.pid" in cmd:
                return RemoteResult(host=host, command=cmd, stdout="\n")  # empty pid
            if "cat /tmp/test_time.out" in cmd:
                return RemoteResult(host=host, command=cmd, stdout="1.0 0.5 50% 1024 0 0\n")
            if "cat /tmp/test_out.json" in cmd:
                return RemoteResult(host=host, command=cmd,
                                    stdout='{"results": {"BW_average": 42, "MsgRate": 0.1}}')
            return RemoteResult(host=host, command=cmd)

        result = self._run(fake)
        self.assertNotIn("error", result)                            # BW preserved
        self.assertEqual(result["results"]["BW_average"], 42)
        self.assertEqual(result["_process"]["server_perf"], {})      # no perf data
        self.assertFalse(any("perf record" in c for c in calls))     # branch skipped
        # ... but the skip is ATTRIBUTED, so the bar renders "n/a" not zero-height
        self.assertIn("perf_start_error", result["_process"])
        self.assertIn("server PID unavailable", result["_process"]["perf_start_error"])

    def test_perf_record_stopped_on_exception_path(self):
        # Regression guard for a perf-record LEAK: if an exception (or Ctrl-C)
        # interrupts the run AFTER `perf record` started but BEFORE the inline
        # SIGINT-stop, perf would otherwise sample on the server forever.  The
        # finally -> _cancel safety net must stop it.  _cancel is deliberately
        # NOT mocked here so its REAL teardown body runs on the exception path
        # (the historical `check=` regression shipped because the cleanup body
        # was mocked away and never executed in the suite).
        calls = []

        def fake(cmd, host, timeout=300, ssh_config=None, sudo=None):
            calls.append(cmd)
            if "cat /tmp/test_server.pid" in cmd:
                return RemoteResult(host=host, command=cmd, stdout="123\n")
            if "cat /tmp/test_perf.pid" in cmd:        # perf pid still recorded
                return RemoteResult(host=host, command=cmd, stdout="456\n")
            if "/usr/bin/time" in cmd:                 # crash mid client run
                raise RuntimeError("SIMULATED crash mid-client-run")
            return RemoteResult(host=host, command=cmd)

        with (
            patch("rdma_sweep._run_remote_result", autospec=True, side_effect=fake),
            patch("rdma_sweep._wait_for_port"),
            # _cancel intentionally NOT patched -> real teardown executes
        ):
            result = run_perftest(
                binary="ib_write_bw",
                server_host="server-ssh",
                client_host="client-ssh",
                server_address="10.0.0.2",
                perftest_config=self._perf_config(),
                ssh_config={"sudo": True, "connect_timeout": 1, "options": []},
                extra_args=["-s", "64K", "-q", "4", "-p", "18515"],
                duration=5,
                use_gpu=False,
            )

        # The run aborts attributing the real cause (not a fake-clean result) ...
        self.assertIn("error", result)
        self.assertIn("SIMULATED crash", result["error"])
        # ... and perf was stopped on the cleanup path, gated on our perf_data.
        perf_teardown = [
            c for c in calls
            if "kill -INT" in c
            and "/tmp/test_perf.data" in c
            and "rm -f /tmp/test_perf.pid" in c
        ]
        self.assertTrue(
            perf_teardown,
            "leaked perf record must be stopped from finally/_cancel on the exception path",
        )

    def test_perf_cleanup_failure_surfaced_at_top_level(self):
        # A perf-record process that survives BOTH SIGINT and SIGKILL keeps
        # sampling+burning CPU on the remote and would skew the next sweep
        # point.  Such a cleanup FAILURE must be surfaced with the same
        # prominence as a server-kill failure (process["perf_cleanup_error"]),
        # not buried one level down in commands.cleanup.perf_error.  _cancel is
        # NOT mocked so the real lift in run_perftest's finally executes.
        def fake(cmd, host, timeout=300, ssh_config=None, sudo=None):
            if "kill -INT" in cmd and "/tmp/test_perf.data" in cmd:
                # The safety-net perf teardown reports it could not kill perf.
                return RemoteResult(host=host, command=cmd, returncode=1,
                                    stderr="perf cleanup failed for pid 456")
            if "cat /tmp/test_server.pid" in cmd:
                return RemoteResult(host=host, command=cmd, stdout="123\n")
            if "cat /tmp/test_perf.pid" in cmd:
                return RemoteResult(host=host, command=cmd, stdout="456\n")
            if "/usr/bin/time" in cmd:
                raise RuntimeError("SIMULATED crash mid-client-run")
            return RemoteResult(host=host, command=cmd)

        with (
            patch("rdma_sweep._run_remote_result", autospec=True, side_effect=fake),
            patch("rdma_sweep._wait_for_port"),
        ):
            result = run_perftest(
                binary="ib_write_bw",
                server_host="server-ssh",
                client_host="client-ssh",
                server_address="10.0.0.2",
                perftest_config=self._perf_config(),
                ssh_config={"sudo": True, "connect_timeout": 1, "options": []},
                extra_args=["-s", "64K", "-q", "4", "-p", "18515"],
                duration=5,
                use_gpu=False,
            )

        process = result["_process"]
        # Failure is surfaced prominently ...
        self.assertIn("perf_cleanup_error", process)
        self.assertTrue(process["perf_cleanup_error"])
        # ... and the raw evidence is still retained one level down.
        self.assertIn("perf_error", process["commands"]["cleanup"])
        # A perf-cleanup failure is NOT a server-kill failure: keep them distinct.
        self.assertNotIn("cleanup_error", process)

    def test_gpu_suffix_appended_to_binary(self):
        """run_perftest(use_gpu=True) appends _gpu suffix to binary in commands."""
        calls = []

        def fake(cmd, host, timeout=300, ssh_config=None, sudo=None):
            calls.append(cmd)
            if "cat /tmp/test_server.pid" in cmd:
                return RemoteResult(host=host, command=cmd,
                                    stdout="123\nMon Jun 22 10:00:00 2026\nrun-1\n")
            if "cat /tmp/test_time.out" in cmd:
                return RemoteResult(host=host, command=cmd,
                                    stdout="1.0 0.5 50% 1024 0 0\n")
            if "cat /tmp/test_out.json" in cmd:
                return RemoteResult(host=host, command=cmd,
                                    stdout='{"results": {"BW_average": 42, "MsgRate": 0.1}}')
            return RemoteResult(host=host, command=cmd)

        # Use perf_record=False so the fake doesn't need perf-handling commands
        cfg = dict(self._perf_config())
        cfg["perf_record"] = False
        result = self._run(fake, use_gpu=True, perf_config=cfg)
        self.assertNotIn("error", result)
        self.assertTrue(
            any("ib_write_bw_gpu" in c for c in calls),
            "GPU suffix must be appended to binary path when use_gpu=True",
        )


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
        commands = []

        # Signature-faithful fake (no **kwargs): any unsupported kwarg raises
        # TypeError inside the suite instead of being silently swallowed.
        def fake_run_remote_result(cmd, host, timeout=300, ssh_config=None, sudo=None):
            commands.append(cmd)
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

        with patch("rdma_sweep._run_remote_result", autospec=True, side_effect=fake_run_remote_result):
            evidence = _cancel(
                "server-ssh",
                {"sudo": True},
                "/tmp/server.pid",
                "ib_write_bw",
                "run-1",
            )

        self.assertIn("pid_read", evidence)
        self.assertIn("cleanup", evidence)
        self.assertIn("cleanup failed", evidence["error"])

        # Guard against re-gutting: the cleanup command must verify process
        # identity before killing (start-time + expected binary) and confirm
        # the process is gone afterwards.  A bare "kill || true; rm -f" can
        # never report a survivor and can kill a process that reused the PID.
        cleanup_cmd = next((c for c in commands if "kill -TERM" in c), None)
        self.assertIsNotNone(cleanup_cmd, "no kill -TERM command issued")
        self.assertIn("lstart", cleanup_cmd)        # identity: start-time match
        self.assertIn("ib_write_bw", cleanup_cmd)   # identity: expected binary
        self.assertIn("ps -p", cleanup_cmd)         # post-kill liveness verification
        self.assertIn("exit 1", cleanup_cmd)        # report a survivor as failure

    def test_cancel_skips_kill_when_run_id_mismatches(self):
        # If the pid file's run-id is not ours, the recorded PID belongs to a
        # different run (or a reused PID) — we must NOT issue any kill command.
        commands = []

        def fake_run_remote_result(cmd, host, timeout=300, ssh_config=None, sudo=None):
            commands.append(cmd)
            if cmd.startswith("cat "):
                return RemoteResult(
                    host=host,
                    command=cmd,
                    stdout="123\nMon Jun 22 10:00:00 2026\nrun-OTHER\n",
                )
            return RemoteResult(host=host, command=cmd)

        with patch("rdma_sweep._run_remote_result", autospec=True, side_effect=fake_run_remote_result):
            evidence = _cancel(
                "server-ssh",
                {"sudo": True},
                "/tmp/server.pid",
                "ib_write_bw",
                "run-1",
            )

        self.assertEqual(len(commands), 1)                 # only the cat, no kill
        self.assertTrue(commands[0].startswith("cat "))
        self.assertIn("pid_read", evidence)
        self.assertNotIn("cleanup", evidence)              # no cleanup attempted
        self.assertNotIn("error", evidence)

    def test_cancel_stops_leaked_perf_record(self):
        # If perf record was left running (an exception/Ctrl-C skipped the
        # inline SIGINT-stop), _cancel must signal it from the finally path —
        # gated on OUR run-specific `-o <perf_data>` in its live cmdline so a
        # reused PID is never killed — and remove its pid file.
        commands = []

        def fake_run_remote_result(cmd, host, timeout=300, ssh_config=None, sudo=None):
            commands.append(cmd)
            return RemoteResult(host=host, command=cmd, stdout="456\n")

        with patch("rdma_sweep._run_remote_result", autospec=True, side_effect=fake_run_remote_result):
            evidence = _cancel(
                "server-ssh",
                {"sudo": True},
                perf_pid_file="/tmp/p.pid",
                perf_data="/tmp/p.data",
            )

        self.assertIn("perf_cleanup", evidence)
        perf_cmd = next((c for c in commands if "kill -INT" in c), None)
        self.assertIsNotNone(perf_cmd, "no kill -INT command issued")
        # Guard against re-gutting (same discipline as the server kill):
        self.assertIn("grep -F", perf_cmd)      # PID-reuse safety: cmdline identity
        self.assertIn("/tmp/p.data", perf_cmd)  # ... on the run-specific target
        self.assertIn("kill -KILL", perf_cmd)   # escalate if INT is ignored
        self.assertIn("ps -p", perf_cmd)        # post-kill liveness verification
        self.assertIn("exit 1", perf_cmd)       # report a survivor as failure
        self.assertIn("rm -f", perf_cmd)        # clear the (now-stale) pid file
        self.assertIn("/tmp/p.pid", perf_cmd)

    def test_cancel_perf_teardown_skipped_without_perf_args(self):
        # Idempotent / safe when perf was never used: with no perf_pid_file the
        # teardown block is a no-op (no extra remote command, no perf evidence).
        commands = []

        def fake_run_remote_result(cmd, host, timeout=300, ssh_config=None, sudo=None):
            commands.append(cmd)
            return RemoteResult(host=host, command=cmd, stdout="123\nMon Jun 22 10:00:00 2026\nrun-1\n")

        with patch("rdma_sweep._run_remote_result", autospec=True, side_effect=fake_run_remote_result):
            evidence = _cancel("server-ssh", {"sudo": True}, "/tmp/server.pid", "ib_write_bw", "run-1")

        self.assertNotIn("perf_cleanup", evidence)
        self.assertFalse(any("/tmp/p.data" in c or "perf cleanup" in c for c in commands))

    def test_cancel_perf_cleanup_exception_stored(self):
        """_cancel exception handler (line 756-757): _run_remote_result raises during
        perf cleanup — evidence["perf_error"] stores the exception string."""
        def fake_raise(cmd, host, timeout=300, ssh_config=None, sudo=None):
            raise RuntimeError("SSH connection lost during perf cleanup")

        with patch("rdma_sweep._run_remote_result", autospec=True, side_effect=fake_raise):
            evidence = _cancel(
                "server-ssh",
                {"sudo": True},
                # No server_pid_file/binary/run_id → return early after perf.
                perf_pid_file="/tmp/p.pid",
                perf_data="/tmp/p.data",
            )

        self.assertIn("perf_error", evidence)
        self.assertIn("SSH connection lost", evidence["perf_error"])
        # perf_cleanup key is absent because .to_dict() was never reached
        self.assertNotIn("perf_cleanup", evidence)

    def test_cancel_pid_read_exception_stored(self):
        """_cancel exception handler (line 793-794): _run_remote_result raises during
        pid_read — evidence["error"] stores the exception string."""
        def fake_raise(cmd, host, timeout=300, ssh_config=None, sudo=None):
            raise RuntimeError("SSH connection lost during pid read")

        with patch("rdma_sweep._run_remote_result", autospec=True, side_effect=fake_raise):
            evidence = _cancel(
                "server-ssh",
                {"sudo": True},
                "/tmp/server.pid",
                "ib_write_bw",
                "run-1",
            )

        self.assertIn("error", evidence)
        self.assertIn("SSH connection lost", evidence["error"])
        # pid_read key is absent because .to_dict() was never reached
        self.assertNotIn("pid_read", evidence)


# ---------------------------------------------------------------------------
# _parse_pid_file_lines  (6 scenarios)
# ---------------------------------------------------------------------------

class TestParsePidFileLines(unittest.TestCase):
    """_parse_pid_file_lines parses the 3-line remote PID file."""

    def test_full_3_line_pid_file(self):
        pid, start, run_id = _parse_pid_file_lines(
            "12345\nMon Jun 22 10:00:00 2026\nabc123def456\n"
        )
        self.assertEqual(pid, "12345")
        self.assertEqual(start, "Mon Jun 22 10:00:00 2026")
        self.assertEqual(run_id, "abc123def456")

    def test_only_pid_line(self):
        pid, start, run_id = _parse_pid_file_lines("98765\n")
        self.assertEqual(pid, "98765")
        self.assertEqual(start, "")
        self.assertEqual(run_id, "")

    def test_empty_input(self):
        pid, start, run_id = _parse_pid_file_lines("")
        self.assertEqual(pid, "")
        self.assertEqual(start, "")
        self.assertEqual(run_id, "")

    def test_extra_lines_ignored(self):
        pid, start, run_id = _parse_pid_file_lines(
            "42\nThu Jan  1 00:00:00 1970\nrun-99\nextra1\nextra2\n"
        )
        self.assertEqual(pid, "42")
        self.assertEqual(start, "Thu Jan  1 00:00:00 1970")
        self.assertEqual(run_id, "run-99")

    def test_whitespace_lines_stripped(self):
        pid, start, run_id = _parse_pid_file_lines("  777  \n  start-time  \n  run-x  \n")
        self.assertEqual(pid, "777")
        self.assertEqual(start, "start-time")
        self.assertEqual(run_id, "run-x")

    def test_two_lines_only(self):
        pid, start, run_id = _parse_pid_file_lines("5555\nWed Jun 24 12:00:00 2026\n")
        self.assertEqual(pid, "5555")
        self.assertEqual(start, "Wed Jun 24 12:00:00 2026")
        self.assertEqual(run_id, "")


class TestPortFromArgs(unittest.TestCase):
    """_port_from_args must recognise every form perftest accepts.

    The polled port (via _wait_for_port) must equal the bound port; otherwise a
    healthy server is mis-reported as "did not listen".  Today _build_args emits
    only ``-p <val>``, but the parser is kept total against future passthrough.
    """

    def test_short_two_token(self):
        self.assertEqual(_port_from_args(["-p", "18515"], 5000), 18515)

    def test_short_attached(self):
        self.assertEqual(_port_from_args(["-p18515"], 5000), 18515)

    def test_long_spaced(self):
        self.assertEqual(_port_from_args(["--port", "18515"], 5000), 18515)

    def test_long_equals(self):
        self.assertEqual(_port_from_args(["--port=18515"], 5000), 18515)

    def test_absent_returns_default(self):
        self.assertEqual(_port_from_args(["-s", "65536", "-q", "4"], 5000), 5000)

    def test_embedded_in_realistic_args(self):
        # mirrors _build_args output order: other flags then the port pair
        args = ["-s", "65536", "-q", "4", "-p", "18515", "-R"]
        self.assertEqual(_port_from_args(args, 5000), 18515)

    def test_dangling_flag_returns_default(self):
        # "-p" with no following value must not IndexError — fall back to default
        self.assertEqual(_port_from_args(["-s", "65536", "-p"], 5000), 5000)

    def test_malformed_value_falls_back_not_crash(self):
        # a non-integer port value must not raise out of the parser mid-sweep
        self.assertEqual(_port_from_args(["--port=notaport"], 5000), 5000)
        self.assertEqual(_port_from_args(["-p", "bad"], 5000), 5000)

    def test_not_confused_by_lookalike_flags(self):
        # -ps / --portal-ish tokens must not be mistaken for a port
        self.assertEqual(_port_from_args(["--portfoo", "1"], 5000), 5000)


class TestWaitForPort(unittest.TestCase):
    """_wait_for_port runs its real body against a signature-faithful fake.

    The run_perftest tests mock _wait_for_port away, so its body never executed
    in the suite. That let a leftover ``check=`` kwarg (a remnant of the removed
    run_remote() wrapper) reach the real run_remote_result() only at runtime:
    every sweep point raised TypeError while all 47 unit tests stayed green.

    The fakes below mirror run_remote_result's exact signature (no ``**kwargs``),
    so any unsupported kwarg raises TypeError inside the suite instead of being
    silently swallowed.
    """

    def test_returns_when_port_is_listening(self):
        calls = []

        def fake_run_remote_result(cmd, host, timeout=300, ssh_config=None, sudo=None):
            calls.append((cmd, host, sudo))
            return RemoteResult(host=host, command=cmd, stdout="ready\n")

        with patch("rdma_sweep._run_remote_result", autospec=True, side_effect=fake_run_remote_result):
            # Must return (no exception) as soon as the probe reports "ready".
            self.assertIsNone(
                _wait_for_port("server-ssh", 18515, timeout=5, ssh_config={"sudo": False})
            )

        self.assertEqual(len(calls), 1)
        cmd, host, sudo = calls[0]
        self.assertEqual(host, "server-ssh")
        self.assertIn(":18515", cmd)
        self.assertFalse(sudo)  # the port probe must not require sudo

    def test_raises_timeout_when_port_never_listens(self):
        def fake_run_remote_result(cmd, host, timeout=300, ssh_config=None, sudo=None):
            return RemoteResult(host=host, command=cmd, stdout="")

        with (
            patch("rdma_sweep._run_remote_result", autospec=True, side_effect=fake_run_remote_result),
            patch("rdma_sweep.time.sleep"),
        ):
            with self.assertRaises(TimeoutError):
                _wait_for_port("server-ssh", 18515, timeout=1, ssh_config={})

    def test_unreachable_host_raises_transport_error_not_listen_error(self):
        # Every probe fails at the SSH/transport layer (non-zero rc, no stdout).
        # The old body looked only at .stdout and would mis-report this as
        # "port did not listen"; the fix must surface the transport failure so
        # the cause (host unreachable) is not mistaken for a missing listener.
        def fake_run_remote_result(cmd, host, timeout=300, ssh_config=None, sudo=None):
            return RemoteResult(
                host=host, command=cmd, returncode=255,
                stderr="ssh: connect to host server-ssh port 22: No route to host",
            )

        with (
            patch("rdma_sweep._run_remote_result", autospec=True, side_effect=fake_run_remote_result),
            patch("rdma_sweep.time.sleep"),
        ):
            with self.assertRaisesRegex(TimeoutError, "cannot reach"):
                _wait_for_port("server-ssh", 18515, timeout=1, ssh_config={})

    def test_reachable_but_not_listening_is_not_a_transport_error(self):
        # A reachable host whose port is simply not up yet: the probe ends in
        # ``|| true`` so .ok is True even though "ready" is absent.  This must
        # raise the *listen* timeout, never the transport-failure message —
        # otherwise we'd swing the mis-attribution the other way.
        def fake_run_remote_result(cmd, host, timeout=300, ssh_config=None, sudo=None):
            return RemoteResult(host=host, command=cmd, stdout="", returncode=0)

        with (
            patch("rdma_sweep._run_remote_result", autospec=True, side_effect=fake_run_remote_result),
            patch("rdma_sweep.time.sleep"),
        ):
            with self.assertRaisesRegex(TimeoutError, "did not listen"):
                _wait_for_port("server-ssh", 18515, timeout=1, ssh_config={})

    def test_ss_missing_short_circuits_with_actionable_error(self):
        # When ``ss`` is not installed on the remote host, the probe stdout
        # contains the ``SS_NOT_FOUND`` sentinel.  The function must raise a
        # clear diagnostic (not "did not listen") and break out of the retry
        # loop instead of burning the full timeout.
        calls = []

        def fake_run_remote_result(cmd, host, timeout=300, ssh_config=None, sudo=None):
            calls.append(cmd)
            return RemoteResult(host=host, command=cmd, stdout="SS_NOT_FOUND\n")

        with (
            patch("rdma_sweep._run_remote_result", autospec=True, side_effect=fake_run_remote_result),
            patch("rdma_sweep.time.sleep"),
        ):
            with self.assertRaisesRegex(TimeoutError, "ss.*not found"):
                _wait_for_port("server-ssh", 18515, timeout=5, ssh_config={})

        # Short-circuit: only 1 probe, not timeout*2 probes.
        self.assertEqual(len(calls), 1)


class TestExpand(unittest.TestCase):
    """_expand turns a sweep-param spec into the concrete list of values.

    The range form uses counter-based arithmetic (``lo + i * step``) to avoid
    float-accumulation drift.  A non-positive step never terminates (step==0)
    or runs away (step<0), and ``hi < lo`` silently yields zero combos — a
    sweep that looks like it ran but tested nothing.  Both are operator typos
    that must fail loud, not hang or no-op, so a bad spec is never mistaken
    for a completed-but-empty sweep.
    """

    def test_list_passthrough_is_copied(self):
        src = [2, 4, 8]
        out = _expand(src)
        self.assertEqual(out, [2, 4, 8])
        out.append(99)
        self.assertEqual(src, [2, 4, 8])  # a copy, not the caller's list

    def test_scalar_wrapped_in_list(self):
        self.assertEqual(_expand(64), [64])

    def test_values_form(self):
        self.assertEqual(_expand({"values": [1, 4, 16]}), [1, 4, 16])

    def test_range_inclusive_of_to(self):
        self.assertEqual(_expand({"from": 1, "to": 4, "step": 1}), [1, 2, 3, 4])

    def test_range_with_step(self):
        self.assertEqual(_expand({"from": 2, "to": 8, "step": 2}), [2, 4, 6, 8])

    def test_range_default_from_zero_default_step_one(self):
        # {"to": 3} alone defaults from=0, step=1 → 0..3 inclusive (qp=0 reachable)
        self.assertEqual(_expand({"to": 3}), [0, 1, 2, 3])

    def test_single_point_from_equals_to(self):
        self.assertEqual(_expand({"from": 4, "to": 4}), [4])

    def test_step_zero_raises_not_hangs(self):
        with self.assertRaises(ValueError):
            _expand({"from": 1, "to": 8, "step": 0})

    def test_negative_step_raises(self):
        with self.assertRaises(ValueError):
            _expand({"from": 1, "to": 8, "step": -1})

    def test_nan_from_raises_finite_error(self):
        """nan 'from' must raise a clear 'finite' error, not an opaque
        ``int(nan)`` ValueError from the range arithmetic."""
        with self.assertRaises(ValueError) as ctx:
            _expand({"from": float("nan"), "to": 8})
        self.assertIn("finite", str(ctx.exception))

    def test_inf_to_raises_finite_error(self):
        """inf 'to' must raise a clear 'finite' error.  Pre-guard this produced
        an ``OverflowError`` (not even a ValueError) from ``int(inf)``."""
        with self.assertRaises(ValueError) as ctx:
            _expand({"from": 1, "to": float("inf")})
        self.assertIn("finite", str(ctx.exception))

    def test_inf_step_raises_not_silent_nan(self):
        """A step of inf must fail loud.  Without the guard it silently
        collapses to a single ``[nan]`` value (0 * inf == nan), injecting
        garbage into the sweep instead of raising."""
        with self.assertRaises(ValueError) as ctx:
            _expand({"from": 1, "to": 8, "step": float("inf")})
        self.assertIn("finite", str(ctx.exception))

    def test_to_less_than_from_raises(self):
        bad = {"from": 8, "to": 1}
        with self.assertRaises(ValueError) as ctx:
            _expand(bad)
        # fail-loud is only actionable if the offending spec is echoed verbatim
        # so the operator can find it; pin that the param is in the message.
        self.assertIn(repr(bad), str(ctx.exception))

    def test_error_message_names_the_offending_spec(self):
        # fail-loud is only useful if the operator can find the bad param
        with self.assertRaises(ValueError) as ctx:
            _expand({"from": 1, "to": 8, "step": 0})
        self.assertIn("step", str(ctx.exception))

    def test_range_dict_without_to_or_values_raises(self):
        # A dict spec must be {values:...} or a {from,to,step} range.  One
        # missing 'to' is a config error: fail loud with the spec echoed,
        # matching the step/to<from guards -- not an opaque KeyError: 'to'.
        bad = {"from": 1, "step": 2}
        with self.assertRaises(ValueError) as ctx:
            _expand(bad)
        self.assertIn(repr(bad), str(ctx.exception))

    def test_empty_values_list_raises(self):
        """A {... 'values': []} dict is a config error: fail loud."""
        bad = {"values": []}
        with self.assertRaises(ValueError) as ctx:
            _expand(bad)
        self.assertIn("empty", str(ctx.exception).lower())

    def test_null_to_raises(self):
        """_expand raises on ``to: null``, not an opaque TypeError."""
        with self.assertRaises(ValueError):
            _expand({"to": None})

    def test_null_from_defaults_to_zero(self):
        """_expand coerces ``from: null`` to 0 so the range stays valid."""
        self.assertEqual(_expand({"from": None, "to": 2}), [0, 1, 2])

    def test_null_step_defaults_to_one(self):
        """_expand coerces ``step: null`` to 1 so the range stays valid."""
        self.assertEqual(_expand({"from": 1, "to": 3, "step": None}), [1, 2, 3])

    def test_bool_from_rejected(self):
        """_expand rejects bool as 'from' — YAML ``true`` is not integer 1."""
        with self.assertRaises(ValueError):
            _expand({"from": True, "to": 4})

    def test_bool_to_rejected(self):
        """_expand rejects ``to: true``, not silently treats it as 1."""
        with self.assertRaises(ValueError):
            _expand({"from": 1, "to": True})

    def test_bool_step_rejected(self):
        """_expand rejects ``step: true`` — bool is not an integer."""
        with self.assertRaises(ValueError):
            _expand({"from": 1, "to": 4, "step": True})

    def test_bool_false_from_rejected(self):
        """_expand rejects False as from even though False == 0 in Python."""
        with self.assertRaises(ValueError):
            _expand({"from": False, "to": 4})

    def test_values_list_contains_null_raises(self):
        """_expand rejects ``values: [1, null, 4]`` before it reaches product()."""
        bad = {"values": [1, None, 4]}
        with self.assertRaises(ValueError) as ctx:
            _expand(bad)
        self.assertIn("null", str(ctx.exception).lower())

    def test_values_list_contains_bool_raises(self):
        """_expand rejects ``values: [1, true, 4]`` — bare YAML ``true`` is a bool,
        not an intentional list element; likely a config error."""
        bad = {"values": [1, True, 4]}
        with self.assertRaises(ValueError) as ctx:
            _expand(bad)
        self.assertIn("bool", str(ctx.exception).lower())

    def test_values_list_contains_false_raises(self):
        """False is a bool; _expand rejects it even though False == 0."""
        with self.assertRaises(ValueError):
            _expand({"values": [0, False, 2]})

    def test_float_step_uses_counter_avoiding_drift(self):
        """_expand uses counter-based arithmetic to avoid float drift."""
        # This would produce [0.1, 0.2] (missing 0.3) with repeated addition
        # due to 0.1+0.1+0.1 > 0.3; counter-based math fixes it.
        r = _expand({"from": 0.1, "to": 0.3, "step": 0.1})
        self.assertEqual(len(r), 3)
        self.assertAlmostEqual(r[0], 0.1)
        self.assertAlmostEqual(r[1], 0.2)
        self.assertAlmostEqual(r[2], 0.3, places=15)


class TestSummaryAttribution(unittest.TestCase):
    """A failed /proc grab must never read as a genuinely idle host.

    grab() returns {"error": ...} on SSH failure, so every cpu/mem diff is
    empty — identical on the surface to a real low-load result.  These tests
    pin the fix that propagates the sys-monitor flags into the summary entry
    (server/client_sys_ok) and renders failed cells as "ERR" (never blank/0)
    in the CSV, so symptom (blank metric) is not mis-attributed to cause
    (idle host) when the real cause is a failed measurement.
    """

    def _idle_meta(self):
        return {
            "server": {"host": "s", "address": "10.0.0.1"},
            "client": {"host": "c"},
            "parameters": {"qp": 1},
            "elapsed_sec": 1.0,
            "cpu_util_per_core": {"cpu0": 0.0, "cpu1": 0.0},
            "server_cpu_util_per_core": {"cpu0": 0.0, "cpu1": 0.0},
            "client_cpu_util_per_core": {"cpu0": 0.0},
            "memory": {"MemUsed": "10.0 GiB", "MemUsedDelta": "0 B"},
            "server_memory": {"MemUsed": "10.0 GiB", "MemUsedDelta": "0 B"},
            "client_memory": {"MemUsed": "8.0 GiB", "MemUsedDelta": "0 B"},
            "server_sys_before_ok": True,
            "server_sys_after_ok": True,
            "client_sys_before_ok": True,
            "client_sys_after_ok": True,
        }

    def _failed_meta(self):
        # post-test grab on the server failed → empty diffs/memory came back
        meta = self._idle_meta()
        meta.update({
            "cpu_util_per_core": {},
            "server_cpu_util_per_core": {},
            "memory": {},
            "server_memory": {},
            "server_sys_after_ok": False,
        })
        return meta

    def test_entry_marks_failed_grab_not_ok(self):
        entry = _summary_entry(self._failed_meta(), {"BW_average": 100})
        self.assertFalse(entry["server_sys_ok"])
        self.assertTrue(entry["client_sys_ok"])  # client side was fine

    def test_entry_marks_clean_grab_ok(self):
        entry = _summary_entry(self._idle_meta(), {"BW_average": 100})
        self.assertTrue(entry["server_sys_ok"])
        self.assertTrue(entry["client_sys_ok"])

    def test_entry_defaults_ok_for_pre_flag_files(self):
        # old result files lack the *_sys_*_ok keys; don't invent a failure
        entry = _summary_entry({"parameters": {}}, {})
        self.assertTrue(entry["server_sys_ok"])
        self.assertTrue(entry["client_sys_ok"])

    def test_entry_with_null_meta_fields_does_not_crash(self):
        """_summary_entry handles null (None) meta fields without crashing."""
        entry = _summary_entry({
            "parameters": None,
            "server_cpu_util_per_core": None,
            "client_cpu_util_per_core": None,
            "memory": None,
            "server_memory": None,
            "client_memory": None,
            "server": None,
            "client": None,
            "server_sys_before_ok": True,
            "server_sys_after_ok": True,
            "client_sys_before_ok": True,
            "client_sys_after_ok": True,
        }, {"BW_average": 100})
        # Must produce valid entry, not crash
        self.assertEqual(entry["server_cpu_avg"], "")
        self.assertEqual(entry["client_cpu_avg"], "")
        self.assertEqual(entry["server"], {})
        self.assertEqual(entry["client"], {})
        self.assertEqual(entry["params"], {})
        self.assertEqual(entry["memory"], {})
        self.assertEqual(entry["server_memory"], {})

    def test_entry_with_null_per_core_values_skips_them_not_crash(self):
        """A present-but-null per-core *value* must be skipped, never crash/0.

        Distinct from test_entry_with_null_meta_fields_does_not_crash above,
        which covers the *whole* per-core dict being null (_get_dict -> {}).
        Here the dict is present with null *values* ({"cpu0": null}), which
        _get_dict passes through untouched, so _cpu_avg's float() would hit
        float(None) and abort the entire report unless it skips null cores --
        the un-fixed _cpu_avg sibling of the qp=null disk-corruption defense
        (mirrors _extract_metric's documented present-None-from-disk guard).
        """
        meta = self._idle_meta()
        # cpu0 came back null (partial/corrupt result.json); cpu1 measured 50.0
        meta["server_cpu_util_per_core"] = {"cpu0": None, "cpu1": 50.0}
        # every client core null -> zero measured cores -> "" (not a crash/0)
        meta["client_cpu_util_per_core"] = {"cpu0": None, "cpu1": None}
        entry = _summary_entry(meta, {"BW_average": 100})
        # honest average over the ONE measured core -- NOT (0+50)/2=25.0, which
        # would masquerade the absent cpu0 as a clean idle 0% (float(v or 0))
        self.assertEqual(entry["server_cpu_avg"], 50.0)
        # all-null degrades to "" like the empty-dict case, never crash or 0.0
        self.assertEqual(entry["client_cpu_avg"], "")

    def test_mem_delta_mib_inf_string_coerces_to_zero_not_crash(self):
        """An inf-shaped MemUsedDelta must coerce to 0.0, not abort the report.

        parse_size("infT") evaluates int(float("inf") * 1024**4) -> Overflow
        Error, which is NOT a ValueError subclass, so it escaped the
        (TypeError, ValueError) guard and aborted the whole report.  Only
        reachable from a corrupt/hand-edited result.json (format_size never
        emits an inf-shaped string), but the guard's evident intent is "any
        malformed MemUsedDelta -> 0.0, never crash"; OverflowError was the hole.
        """
        self.assertEqual(_mem_delta_mib({"m": {"MemUsedDelta": "infT"}}, "m"), 0.0)
        # sanity: a well-formed delta still parses (guard didn't swallow data)
        self.assertEqual(_mem_delta_mib({"m": {"MemUsedDelta": "1M"}}, "m"), 1.0)

    def test_core_cell_present_null_value_renders_na_not_crash(self):
        """The per-core *table* sibling of _cpu_avg: present-null -> "n/a".

        _svg_chart consumes the same disk-sourced per-core dict on BOTH the
        average path (_cpu_avg) and the table path (_core_cell) in one report
        build, so a corrupt/partial result.json carrying {"cpu0": null} must
        not abort here either.  The old `core in per_core` test guarded key
        *absence* only, so a present-null value reached f"{None:.1f}" and
        crashed the whole report.  "n/a" (not 0.0) keeps it honest; a genuine
        idle 0.0 must still render "0.0" (a real reading is never shown absent).
        """
        # present-null value -> "n/a" (absent-measurement sentinel, never 0.0)
        self.assertEqual(_core_cell({"cpu0": None, "cpu1": 50.0}, "cpu0"), "n/a")
        # a real value alongside it still renders (guard didn't swallow data)
        self.assertEqual(_core_cell({"cpu0": None, "cpu1": 50.0}, "cpu1"), "50.0")
        # GENUINE idle 0.0% must stay "0.0", NOT become "n/a" (inverse masquerade)
        self.assertEqual(_core_cell({"cpu0": 0.0}, "cpu0"), "0.0")
        # absent core still "n/a" (pre-existing union-gap behavior preserved)
        self.assertEqual(_core_cell({"cpu0": 50.0}, "cpu9"), "n/a")

    def test_entry_with_null_results_returns_no_metrics(self):
        """_summary_entry handles results=None without crashing or injecting stale metrics."""
        entry = _summary_entry({
            "parameters": {"qp": 8},
            "server": {"host": "h1"},
            "client": {"host": "h2"},
            "server_sys_before_ok": True,
            "server_sys_after_ok": True,
            "client_sys_before_ok": True,
            "client_sys_after_ok": True,
        }, None)
        # Meta fields still propagate
        self.assertEqual(entry["params"], {"qp": 8})
        self.assertEqual(entry["server"]["host"], "h1")
        self.assertTrue(entry["server_sys_ok"])
        self.assertTrue(entry["client_sys_ok"])
        # No perftest metrics — results was None
        self.assertNotIn("BW_average", entry)
        self.assertNotIn("MsgRate", entry)
        self.assertNotIn("t_avg", entry)
        # Error also absent (there was no run_error in meta)
        self.assertIsNone(entry.get("error"))

    def _csv_rows(self, summary):
        import csv
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "summary.csv"
            _write_csv(path, summary)
            with open(path, newline="") as f:
                return list(csv.reader(f))

    def test_csv_distinguishes_failed_grab_from_idle(self):
        idle = _summary_entry(self._idle_meta(), {"BW_average": 100})
        failed = _summary_entry(self._failed_meta(), {"BW_average": 100})
        rows = self._csv_rows([idle, failed])
        header = rows[0]
        self.assertIn("server_sys_ok", header)
        self.assertIn("client_sys_ok", header)
        idx = {name: i for i, name in enumerate(header)}
        idle_row, failed_row = rows[1], rows[2]

        # idle: real number + ok flag, never the ERR sentinel
        self.assertEqual(idle_row[idx["server_sys_ok"]], "True")
        self.assertEqual(idle_row[idx["server_cpu_avg"]], "0.0")
        # The mem-used / mem-used-delta columns are derived from server_memory
        # directly (NOT from any precomputed entry key) -- pin the real values
        # so a future "unify" refactor that re-points the column at a
        # removed/empty entry key can't silently blank it.
        self.assertEqual(idle_row[idx["server_mem_used"]], "10.0 GiB")
        self.assertEqual(idle_row[idx["server_mem_used_delta"]], "0 B")

        # failed: sentinel everywhere the server metric would otherwise look idle
        self.assertEqual(failed_row[idx["server_sys_ok"]], "False")
        self.assertEqual(failed_row[idx["server_cpu_avg"]], "ERR")
        self.assertEqual(failed_row[idx["cpu_avg"]], "ERR")
        self.assertEqual(failed_row[idx["server_mem_used"]], "ERR")
        self.assertEqual(failed_row[idx["server_mem_used_delta"]], "ERR")

        # the client side of the failed run was fine — must NOT be poisoned
        self.assertEqual(failed_row[idx["client_sys_ok"]], "True")
        self.assertNotEqual(failed_row[idx["client_cpu_avg"]], "ERR")

        # the whole point: an idle row and a failed row are NOT interchangeable
        self.assertNotEqual(
            idle_row[idx["server_cpu_avg"]], failed_row[idx["server_cpu_avg"]]
        )

    # --- errored-run BW masquerade ---------------------------------------
    # A perftest run that wrote valid JSON and THEN exited non-zero (or whose
    # metrics failed validation) carries both ``error`` and a stale BW_average.
    # run_perftest layers the error on top of the parsed result, so the summary
    # entry keeps the raw number for forensics — but the CSV/SVG must never
    # render it as a clean data point.  Same trust bar as the sys-monitor path.

    def _errored_run_meta(self):
        # sys grabs were clean; only the perftest RUN failed (orthogonal axes)
        meta = self._idle_meta()
        meta["run_error"] = "client run failed: exit 2"
        return meta

    def test_entry_keeps_raw_bw_but_flags_run_error(self):
        entry = _summary_entry(self._errored_run_meta(), {"BW_average": 88888, "MsgRate": 77})
        # raw value retained in the structured summary for forensics ...
        self.assertEqual(entry["BW_average"], 88888)
        # ... but the run is unambiguously flagged as errored
        self.assertTrue(entry["error"])
        # the run error must NOT poison the (clean) sys-monitor flags
        self.assertTrue(entry["server_sys_ok"])
        self.assertTrue(entry["client_sys_ok"])

    def test_csv_errored_run_bw_is_err_not_stale_number(self):
        good = _summary_entry(self._idle_meta(), {"BW_average": 200, "MsgRate": 5})
        errored = _summary_entry(self._errored_run_meta(), {"BW_average": 88888, "MsgRate": 77})
        rows = self._csv_rows([good, errored])
        header = rows[0]
        idx = {name: i for i, name in enumerate(header)}
        good_row, err_row = rows[1], rows[2]

        # good run: the real bandwidth/rate numbers are rendered verbatim
        self.assertEqual(good_row[idx["BW_average"]], "200")
        self.assertEqual(good_row[idx["MsgRate"]], "5")

        # errored run: every perf cell is the ERR sentinel, never a clean number
        self.assertEqual(err_row[idx["BW_average"]], "ERR")
        self.assertEqual(err_row[idx["MsgRate"]], "ERR")
        self.assertEqual(err_row[idx["BW_peak"]], "ERR")
        # the stale magnitude must not appear ANYWHERE in the row
        self.assertNotIn("88888", err_row)
        # the error column still explains why
        self.assertTrue(err_row[idx["error"]])

        # the run error is orthogonal to the sys-monitor path — those columns,
        # which were clean, must stay clean (not collateral "ERR").
        self.assertEqual(err_row[idx["server_sys_ok"]], "True")
        self.assertEqual(err_row[idx["server_cpu_avg"]], "0.0")

    def test_svg_omits_errored_run_from_perf_curve(self):
        good = _summary_entry(self._idle_meta(), {"BW_average": 200, "MsgRate": 5})
        errored = _summary_entry(self._errored_run_meta(), {"BW_average": 88888, "MsgRate": 77})
        svg = _svg_chart([good, errored])
        # the failed run's stale magnitudes must never be plotted as points
        self.assertNotIn("88888", svg)          # stale BW_average label
        self.assertNotIn("77000", svg)          # stale MsgRate label (rate * 1000)
        # there IS a valid run, so the chart is drawn (not the empty-state note)
        self.assertNotIn("no valid runs", svg)

    def test_svg_all_runs_errored_shows_no_valid_runs(self):
        errored = _summary_entry(self._errored_run_meta(), {"BW_average": 11111, "MsgRate": 9})
        svg = _svg_chart([errored])
        # nothing valid to plot → explicit empty-state, never a fabricated point
        self.assertIn("no valid runs", svg)
        self.assertNotIn("11111", svg)

    # --- failed /proc grab masquerade in the SVG (sibling of the CSV fix) ---
    # The CSV renders a failed grab as "ERR"; the SVG must not plot it as a
    # clean 0.0% CPU line or a 0.0 per-core table cell.  Server and client carry
    # independent sys_ok flags, so each host's series is filtered on its own.

    def _busy_server_meta(self):
        meta = self._idle_meta()
        meta["parameters"] = {"qp": 1}
        meta["server_cpu_util_per_core"] = {"cpu0": 37.0, "cpu1": 47.0}  # avg 42.0
        meta["cpu_util_per_core"] = {"cpu0": 37.0, "cpu1": 47.0}
        return meta

    def _failed_grab_meta(self, qp):
        meta = self._failed_meta()  # server grab failed → server_sys_ok False
        meta["parameters"] = {"qp": qp}
        return meta

    def test_svg_omits_failed_grab_from_cpu_series_and_table(self):
        busy = _summary_entry(self._busy_server_meta(), {"BW_average": 200, "MsgRate": 5})
        failed = _summary_entry(self._failed_grab_meta(2), {"BW_average": 200, "MsgRate": 5})
        svg = _svg_chart([busy, failed])
        # server CPU line (#dc2626): only the good run is plotted; the failed
        # grab must NOT appear as a clean 0.0 idle reading.
        self.assertEqual(svg.count("r='3.5' fill='#dc2626'"), 1)
        # per-core table: the failed-grab row is "ERR" per core, never 0.0.
        self.assertEqual(svg.count(">ERR</text>"), 2)  # cpu0 + cpu1
        # the client side was fine on the failed run — its series stays intact
        # (#0891b2 = client cpu), so BOTH runs plot a client point.
        self.assertEqual(svg.count("r='3.5' fill='#0891b2'"), 2)

    def test_svg_all_server_grabs_failed_yields_no_fake_server_points(self):
        f1 = _summary_entry(self._failed_grab_meta(1), {"BW_average": 200})
        f2 = _summary_entry(self._failed_grab_meta(2), {"BW_average": 200})
        svg = _svg_chart([f1, f2])
        # every server grab failed → no fabricated server cpu point ...
        self.assertEqual(svg.count("r='3.5' fill='#dc2626'"), 0)
        # ... but the (still-ok) client side keeps plotting both points.
        self.assertEqual(svg.count("r='3.5' fill='#0891b2'"), 2)

    # --- errored run in the "Top CPU Consumers" stacked bar (sibling of the ---
    # bw/rate curve fix).  run_perftest writes process["server_perf"] BEFORE it
    # layers result["error"], so a failed run carries a populated server profile
    # on disk.  The stacked bar reads that profile from result.json, so it must
    # be gated on the same run-error flag — a failed run's CPU profile must never
    # be charted as a valid measurement.

    def _write_result(self, tmp, idx, server_perf):
        p = Path(tmp) / f"{idx:04d}" / "result.json"
        p.parent.mkdir(parents=True)
        p.write_text(json.dumps({"_process": {"server_perf": server_perf}}))
        return str(p)

    def test_svg_omits_errored_run_from_top_cpu_consumers(self):
        good = _summary_entry(self._idle_meta(), {"BW_average": 200, "MsgRate": 5})
        errored = _summary_entry(self._errored_run_meta(), {"BW_average": 88888})
        with tempfile.TemporaryDirectory() as tmp:
            good["_result_path"] = self._write_result(tmp, 1, {"good_sym_clean": 60.0})
            errored["_result_path"] = self._write_result(tmp, 2, {"bogus_sym_errored": 90.0})
            svg = _svg_chart([good, errored])
        # the clean run's profile IS charted ...
        self.assertIn("good_sym_clean", svg)
        # ... but the errored run's profile must NOT be aggregated into the chart
        self.assertNotIn("bogus_sym_errored", svg)

    def test_svg_all_errored_top_cpu_consumers_no_crash(self):
        # every run errored → the stacked bar has zero valid columns; it must
        # render an empty-state note, not crash on a zero-width bar (cw / n, n=0).
        errored = _summary_entry(self._errored_run_meta(), {"BW_average": 11111})
        with tempfile.TemporaryDirectory() as tmp:
            errored["_result_path"] = self._write_result(tmp, 1, {"bogus_sym_errored": 90.0})
            svg = _svg_chart([errored])  # must not raise
        self.assertNotIn("bogus_sym_errored", svg)
        self.assertIn("no valid runs", svg)   # empty-state note, not a blank panel

    def test_svg_null_perf_self_pct_does_not_crash(self):
        # A corrupt or hand-edited result.json can carry a null self-% for a
        # symbol.  _load_perf_bar_series sorts by ``(v or 0)`` but once compared
        # the raw ``None`` with ``> 0`` directly, so a single null entry raised
        # ``TypeError: '>' not supported between 'NoneType' and 'int'`` and
        # aborted the whole report.  The null symbol must be skipped while the
        # valid sibling still charts.
        good = _summary_entry(self._idle_meta(), {"BW_average": 200, "MsgRate": 5})
        with tempfile.TemporaryDirectory() as tmp:
            good["_result_path"] = self._write_result(
                tmp, 1, {"null_sym": None, "real_sym_charted": 70.0})
            svg = _svg_chart([good])  # must not raise on the null self-%
        self.assertIn("real_sym_charted", svg)
        self.assertNotIn("null_sym", svg)

    def test_svg_failed_profile_in_ok_run_renders_na_not_zero_bar(self):
        # An OK run (valid BW, in ok_idx) whose perf profile was ATTEMPTED but
        # failed to collect carries no server_perf — only perf_collection_error
        # (or perf_start_error).  run_perftest now preserves such a run's valid
        # BW instead of aborting it, so it reaches the stacked bar.  Its column
        # must render "n/a" (profile unavailable), never a zero-height bar that
        # masquerades as a clean "no CPU consumers" reading.  Sibling to the
        # corrupt/missing-file None sentinel; here the file is valid JSON, so the
        # gate is the error key, not the read failure.
        good = _summary_entry(self._idle_meta(), {"BW_average": 200, "MsgRate": 5})
        failed = _summary_entry(self._idle_meta(), {"BW_average": 180, "MsgRate": 4})
        with tempfile.TemporaryDirectory() as tmp:
            good["_result_path"] = self._write_result(tmp, 1, {"hot_sym_clean": 60.0})
            # a readable, valid-JSON result.json whose profile failed to collect
            fp = Path(tmp) / "0002" / "result.json"
            fp.parent.mkdir(parents=True)
            fp.write_text(json.dumps(
                {"_process": {"perf_collection_error": "perf stop failed: timeout"}}))
            failed["_result_path"] = str(fp)
            svg = _svg_chart([good, failed])
        # the good run's real profile IS charted ...
        self.assertIn("hot_sym_clean", svg)
        self.assertNotIn("no valid runs", svg)
        # ... and the failed-profile run is the SOLE source of n/a here (both runs
        # share identical default per-core data, so the table fabricates none):
        # exactly one bar column annotated unavailable, not a hidden zero-height bar.
        self.assertEqual(svg.count(">n/a</text>"), 1)

    def test_svg_start_failed_profile_in_ok_run_renders_na_not_zero_bar(self):
        # Sibling of the test above for the OTHER half of the n/a gate: a run
        # whose perf START failed (perf_start_error, e.g. PID unreadable or the
        # ``perf record`` command itself failing) carries no server_perf.  The
        # gate keys on perf_start_error OR perf_collection_error, so this run must
        # also render "n/a", never a zero-height "no CPU consumers" bar.  Without
        # the perf_start_error half of the gate this run would silently render as
        # a clean empty bar (the masquerade _start_perf_record now attributes).
        good = _summary_entry(self._idle_meta(), {"BW_average": 200, "MsgRate": 5})
        failed = _summary_entry(self._idle_meta(), {"BW_average": 180, "MsgRate": 4})
        with tempfile.TemporaryDirectory() as tmp:
            good["_result_path"] = self._write_result(tmp, 1, {"hot_sym_clean": 60.0})
            fp = Path(tmp) / "0002" / "result.json"
            fp.parent.mkdir(parents=True)
            fp.write_text(json.dumps(
                {"_process": {"perf_start_error": "perf requested but server PID unavailable"}}))
            failed["_result_path"] = str(fp)
            svg = _svg_chart([good, failed])
        self.assertIn("hot_sym_clean", svg)
        self.assertNotIn("no valid runs", svg)
        self.assertEqual(svg.count(">n/a</text>"), 1)

    # --- non-positive qp on the log2 x-axis (sweep default from=0) ----------
    # _expand defaults a {"to": N} sweep to from=0, so qp=0 is a reachable run.
    # The bw/rate (_line_chart) and cpu/mem (_multi_line_chart) panels place x
    # with math.log2(qp), which raises on qp<=0 and would crash the whole report
    # mid-render.  Non-positive x must be dropped from the axis, leaving the
    # positive runs plotted — never a crash, never a fabricated point at 0.

    def test_svg_qp_zero_does_not_crash_and_keeps_positive_points(self):
        good = self._idle_meta()
        good["parameters"] = {"qp": 2}
        zero = self._idle_meta()
        zero["parameters"] = {"qp": 0}
        e_good = _summary_entry(good, {"BW_average": 200, "MsgRate": 5})
        e_zero = _summary_entry(zero, {"BW_average": 150, "MsgRate": 3})
        svg = _svg_chart([e_good, e_zero])  # must not raise (no math.log2(0))
        # the positive-qp run is still plotted on bw (#2563eb) and rate (#16a34a)
        # curves; the qp=0 point is silently dropped, not crashed on.
        self.assertEqual(svg.count("r='4' fill='#2563eb'"), 1)
        self.assertEqual(svg.count("r='4' fill='#16a34a'"), 1)
        # there IS valid positive data → real chart, not the empty-state note.
        self.assertNotIn("no valid runs", svg)

    def test_svg_all_qp_zero_shows_no_valid_runs_not_crash(self):
        # a degenerate sweep where every run landed on qp=0: nothing is
        # representable on a log2 axis, so each panel shows the empty-state note
        # rather than raising.
        z1 = self._idle_meta(); z1["parameters"] = {"qp": 0}
        z2 = self._idle_meta(); z2["parameters"] = {"qp": 0}
        e1 = _summary_entry(z1, {"BW_average": 200, "MsgRate": 5})
        e2 = _summary_entry(z2, {"BW_average": 150, "MsgRate": 3})
        svg = _svg_chart([e1, e2])  # must not raise
        self.assertIn("no valid runs", svg)
        # no bw/rate point was fabricated at the unrepresentable x=0
        self.assertEqual(svg.count("r='4' fill='#2563eb'"), 0)

    def test_svg_qp_none_does_not_crash_indexes_positionally(self):
        # A result.json can carry params.qp = null (YAML null / hand-edit).  The
        # series builder must map a present-but-null qp to its positional index
        # (i+1), NOT pass it to float().  The crux: dict.get("qp", i+1) returns
        # the default only when the key is ABSENT -- a present None reaches
        # float(None) and raises TypeError, aborting the whole report.  The
        # None-walrus guard handles it; this asserts the null run still plots.
        good = self._idle_meta(); good["parameters"] = {"qp": 4}
        null_qp = self._idle_meta(); null_qp["parameters"] = {"qp": None}
        e_good = _summary_entry(good, {"BW_average": 200, "MsgRate": 5})
        e_null = _summary_entry(null_qp, {"BW_average": 150, "MsgRate": 3})
        svg = _svg_chart([e_good, e_null])  # must not raise (no float(None))
        # both runs plot: qp=4 at log2(4); null qp mapped to positional i+1=2 at log2(2)
        self.assertEqual(svg.count("r='4' fill='#2563eb'"), 2)
        self.assertNotIn("no valid runs", svg)

    # --- failed /proc grab on the FIRST run must not erase the whole table --
    # The per-core table once derived its column set from summary[0] alone.  If
    # the first run's grab failed its per-core dict is empty, so that oracle
    # produced zero columns and erased EVERY later run's valid per-core data — a
    # genuine measurement rendered as absent (the inverse of the masquerade).
    # Columns now come from the UNION of all runs, so a failed-first run leaves
    # the table intact with later runs' real numbers and only its own row "ERR".

    def test_svg_failed_grab_first_keeps_later_runs_core_table(self):
        failed_first = _summary_entry(self._failed_grab_meta(1), {"BW_average": 200})
        busy = _summary_entry(self._busy_server_meta(), {"BW_average": 200})
        svg = _svg_chart([failed_first, busy])
        # the table is NOT erased: both core columns survive from the union ...
        self.assertIn(">cpu0</text>", svg)
        self.assertIn(">cpu1</text>", svg)
        # ... the good (second) run's real per-core numbers are charted ...
        self.assertIn(">37.0</text>", svg)
        self.assertIn(">47.0</text>", svg)
        # ... and the failed-first run's own row is ERR per core (cpu0 + cpu1),
        # never a fabricated 0.0 idle reading.
        self.assertEqual(svg.count(">ERR</text>"), 2)

    def test_svg_absent_core_in_ok_run_renders_na_not_fabricated_zero(self):
        # Columns are the UNION of all runs.  If a SUCCESSFUL run's grab simply
        # lacks a core another run has (a cpu offline / hot-plugged mid-sweep),
        # that union column must not be back-filled with a fabricated 0.0% idle
        # reading for the run that never measured it — the inverse masquerade
        # (absent data shown as a clean measurement).  It renders "n/a" instead,
        # distinct from "ERR" (which means the whole grab failed).
        narrow = self._idle_meta()
        narrow["parameters"] = {"qp": 1}
        narrow["server_cpu_util_per_core"] = {"cpu0": 10.0, "cpu1": 20.0}
        wide = self._idle_meta()
        wide["parameters"] = {"qp": 2}
        wide["server_cpu_util_per_core"] = {"cpu0": 30.0, "cpu1": 40.0, "cpu2": 50.0}
        e_narrow = _summary_entry(narrow, {"BW_average": 100})
        e_wide = _summary_entry(wide, {"BW_average": 200})
        svg = _svg_chart([e_narrow, e_wide])
        # the union surfaces cpu2 as a column, and the wide run's real number ...
        self.assertIn(">cpu2</text>", svg)
        self.assertIn(">50.0</text>", svg)
        # ... but the narrow run, which never measured cpu2, shows the absent
        # marker in the per-core table.  The stacked-bar panel also emits n/a
        # annotations for runs whose result.json is absent (no temp files here),
        # so the total count is > 1; assert at least one to cover the table case.
        self.assertGreaterEqual(svg.count(">n/a</text>"), 1)
        # both grabs succeeded → no ERR anywhere (n/a and ERR are distinct cases)
        self.assertNotIn(">ERR</text>", svg)

    # --- latency metric surfacing (t_avg was validated then dropped) --------

    def test_csv_surfaces_latency_t_avg(self):
        # a successful *_lat run produces t_avg but no BW_average/MsgRate
        lat = _summary_entry(self._idle_meta(), {"t_avg": 1.23})
        rows = self._csv_rows([lat])
        header = rows[0]
        self.assertIn("t_avg", header)  # column exists ...
        idx = {name: i for i, name in enumerate(header)}
        self.assertEqual(rows[1][idx["t_avg"]], "1.23")  # ... and is populated
        # BW columns are simply empty for a latency test (no fabricated 0)
        self.assertEqual(rows[1][idx["BW_average"]], "")

    def test_csv_errored_latency_t_avg_is_err(self):
        lat = _summary_entry(self._errored_run_meta(), {"t_avg": 1.23})
        rows = self._csv_rows([lat])
        idx = {name: i for i, name in enumerate(rows[0])}
        # latency metric is gated on error exactly like the bandwidth metrics
        self.assertEqual(rows[1][idx["t_avg"]], "ERR")

    def test_svg_latency_sweep_plots_t_avg_not_fake_zero_throughput(self):
        # A *_lat sweep's successful runs carry t_avg (us) and NO BW_average/
        # MsgRate.  The SVG (the tool's headline output) once hardcoded throughput
        # panels, so a valid latency run coerced to BW=0 was drawn as a flat-zero
        # bandwidth curve with the latency number shown NOWHERE — a real
        # measurement displayed as absent (the inverse masquerade).  The chart must
        # now plot t_avg and must NOT fabricate a zero throughput curve.
        n = self._idle_meta(); n["parameters"] = {"qp": 1}
        w = self._idle_meta(); w["parameters"] = {"qp": 2}
        e1 = _summary_entry(n, {"t_avg": 2.5})
        e2 = _summary_entry(w, {"t_avg": 3.75})
        # precondition for the latency-detect branch: t_avg present, BW absent
        self.assertIn("t_avg", e1)
        self.assertNotIn("BW_average", e1)
        svg = _svg_chart([e1, e2])  # must not raise
        # the latency panel is present and BOTH t_avg values are charted with real
        # precision (dec=2 — single-digit us must not round to a meaningless int) ...
        self.assertIn("Latency t_avg (us)", svg)
        self.assertIn(">2.50</text>", svg)
        self.assertIn(">3.75</text>", svg)
        # ... a purple latency data point exists (proves the curve was drawn) ...
        self.assertIn("r='4' fill='#7c3aed'", svg)
        # ... the throughput panels are NOT drawn (no fabricated zero curve, and
        # no blue/green bw/rate data point) ...
        self.assertNotIn("Bandwidth (MB/s)", svg)
        self.assertNotIn("Message Rate (Mmsg/s)", svg)
        self.assertNotIn("r='4' fill='#2563eb'", svg)
        self.assertNotIn("r='4' fill='#16a34a'", svg)
        # ... and the missing throughput is stated explicitly, not left blank.
        self.assertIn("N/A for latency test", svg)

    def test_svg_all_errored_latency_sweep_titled_latency_not_bandwidth(self):
        # When EVERY run of a *_lat sweep errored, ok_idx is empty so the metric
        # family can't be read from successful runs — but the errored entries still
        # carry t_avg.  The headline SVG must title this "Latency ... no valid runs"
        # (honest empty state for the right metric), NOT mis-title it "Bandwidth"
        # (a failed latency sweep masquerading as a failed throughput sweep).
        e1 = _summary_entry(self._errored_run_meta(), {"t_avg": 2.5})
        e2 = _summary_entry(self._errored_run_meta(), {"t_avg": 3.75})
        self.assertTrue(e1.get("error"))  # precondition: both runs errored ...
        self.assertIn("t_avg", e1)        # ... yet still carry the latency metric
        svg = _svg_chart([e1, e2])  # must not raise
        # titled for the correct metric family, with the honest empty state ...
        self.assertIn("Latency t_avg (us)", svg)
        self.assertIn("no valid runs", svg)
        # ... and NOT mislabeled as a throughput sweep, nor a fabricated curve.
        self.assertNotIn("Bandwidth (MB/s)", svg)
        self.assertNotIn("Message Rate (Mmsg/s)", svg)
        self.assertNotIn("r='4' fill='#7c3aed'", svg)  # no plotted point (all errored)


class TestSysOk(unittest.TestCase):
    """Tests for _sys_ok — AND of before+after system-monitor flags."""

    def test_server_both_true(self):
        self.assertTrue(_sys_ok({"server_sys_before_ok": True, "server_sys_after_ok": True}, "server"))

    def test_client_both_true(self):
        self.assertTrue(_sys_ok({"client_sys_before_ok": True, "client_sys_after_ok": True}, "client"))

    def test_server_before_false(self):
        self.assertFalse(_sys_ok({"server_sys_before_ok": False, "server_sys_after_ok": True}, "server"))

    def test_server_after_false(self):
        self.assertFalse(_sys_ok({"server_sys_before_ok": True, "server_sys_after_ok": False}, "server"))

    def test_both_false(self):
        self.assertFalse(_sys_ok({"server_sys_before_ok": False, "server_sys_after_ok": False}, "server"))

    def test_defaults_true(self):
        self.assertTrue(_sys_ok({}, "server"))

    def test_irrelevant_keys_ignored(self):
        self.assertTrue(_sys_ok({"some_other_key": "foo"}, "client"))

    def test_truthy_strings(self):
        self.assertTrue(_sys_ok({"server_sys_before_ok": "yes", "server_sys_after_ok": "ok"}, "server"))

    def test_side_independence(self):
        """server flags should not affect client result and vice versa."""
        meta = {"server_sys_before_ok": False, "server_sys_after_ok": True,
                "client_sys_before_ok": True, "client_sys_after_ok": False}
        self.assertFalse(_sys_ok(meta, "server"))
        self.assertFalse(_sys_ok(meta, "client"))
        # Flip server: now server OK, client still not OK
        meta["server_sys_before_ok"] = True
        self.assertTrue(_sys_ok(meta, "server"))
        self.assertFalse(_sys_ok(meta, "client"))


class TestReportGeneration(unittest.TestCase):
    """Report generation fails clearly for invalid summaries."""

    def test_empty_summary_fails_clearly(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "summary.json").write_text("[]")
            with self.assertRaisesRegex(ValueError, "no sweep entries"):
                generate_report(str(out))

    def test_happy_path_writes_chart_svg(self):
        """generate_report writes chart.svg for a minimal valid summary."""
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            entry = {
                "BW_average": 25000.0,
                "MsgRate": 3.8,
                "params": {"qp": 4},
                "server_cpu_per_core": {"cpu0": 42.0},
            }
            (out / "summary.json").write_text(json.dumps([entry]))
            # result.json stub: _svg_chart reads _process.server_perf from it
            run_dir = out / "0001"
            run_dir.mkdir()
            (run_dir / "result.json").write_text(json.dumps({
                "_process": {"server_perf": {"run_iterations@ib_write_bw": 35.2}}
            }))

            generate_report(str(out))

            svg_path = out / "chart.svg"
            self.assertTrue(svg_path.exists(), "chart.svg was not written")
            svg = svg_path.read_text()
            self.assertIn("<svg", svg)
            self.assertIn("RDMA Perftest Sweep", svg)
            # The _result_path injection + on-disk profile read are the untested
            # glue this e2e test exists to cover.  Without this read the stacked
            # bar chart would have no symbols; assert one is present.
            self.assertIn("run_iterations", svg)

    def test_null_process_in_result_json_does_not_crash(self):
        """#9: generate_report handles result.json with _process: null."""
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            entry = {
                "BW_average": 25000.0,
                "MsgRate": 3.8,
                "params": {"qp": 4},
                "server_cpu_per_core": {"cpu0": 42.0},
            }
            (out / "summary.json").write_text(json.dumps([entry]))
            run_dir = out / "0001"
            run_dir.mkdir()
            # result.json has _process: null — this was crashing _svg_chart
            (run_dir / "result.json").write_text(json.dumps({
                "_process": None
            }))

            try:
                generate_report(str(out))
            except Exception:
                self.fail("generate_report() crashed on _process: null")

            svg_path = out / "chart.svg"
            self.assertTrue(svg_path.exists(), "chart.svg was not written")

    def test_write_csv_empty_summary_no_op(self):
        """_write_csv does not create file for empty summary list."""
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "nope.csv"
            _write_csv(csv_path, [])
            self.assertFalse(csv_path.exists(), "_write_csv must not create file for empty summary")


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

    def test_zero_total_delta_returns_zero(self):
        b = {"cores": {"cpu0": {"user": 1000, "nice": 0, "system": 500, "idle": 50000, "iowait": 0}}}
        a = {"cores": {"cpu0": {"user": 1000, "nice": 0, "system": 500, "idle": 50000, "iowait": 0}}}
        r = SysMonitor.compute_cpu_diff(b, a)
        self.assertEqual(r["cpu0"], 0.0)

    def test_extra_core_in_before_skipped(self):
        """A core in 'before' but missing from 'after' is skipped (continue on line 204)."""
        b = {"cores": {
            "cpu0": {"user": 1000, "nice": 0, "system": 500, "idle": 50000, "iowait": 0},
            "cpu1": {"user": 2000, "nice": 0, "system": 600, "idle": 60000, "iowait": 0},
        }}
        a = {"cores": {
            "cpu0": {"user": 1050, "nice": 0, "system": 530, "idle": 50010, "iowait": 0},
            # cpu1 intentionally missing — should be skipped, not crash
        }}
        r = SysMonitor.compute_cpu_diff(b, a)
        self.assertIn("cpu0", r)
        self.assertNotIn("cpu1", r)
        # cpu0 delta should be computed normally
        self.assertGreater(r["cpu0"], 85.0)
        self.assertLess(r["cpu0"], 95.0)

    def test_extra_core_in_after_omitted_silently(self):
        """A core in 'after' but missing from 'before' is omitted (iterates be keys)."""
        b = {"cores": {
            "cpu0": {"user": 1000, "nice": 0, "system": 500, "idle": 50000, "iowait": 0},
        }}
        a = {"cores": {
            "cpu0": {"user": 1050, "nice": 0, "system": 530, "idle": 50010, "iowait": 0},
            "cpu1": {"user": 2000, "nice": 0, "system": 600, "idle": 60000, "iowait": 0},
        }}
        r = SysMonitor.compute_cpu_diff(b, a)
        self.assertIn("cpu0", r)
        self.assertNotIn("cpu1", r)


class TestSysMonitorGrab(unittest.TestCase):
    """grab() must report remote failure, not return a falsely-'ok' snapshot.

    run_remote_result never raises, so _run() detects ``not result.ok`` itself.
    Without that, a dead host yields empty cores/mem/softirqs with no "error"
    key, and run_sweep's ``*_sys_*_ok`` flags would lie about monitoring.
    """

    def test_grab_reports_error_on_remote_failure(self):
        def fake_run_remote_result(cmd, host, timeout=300, ssh_config=None, sudo=None):
            return RemoteResult(
                host=host, command=cmd, returncode=255,
                stderr="ssh: connect to host failed",
            )

        with patch("rdma_sweep._run_remote_result", autospec=True,
                   side_effect=fake_run_remote_result):
            snapshot = SysMonitor("dead-host").grab()

        self.assertIn("error", snapshot)
        self.assertNotIn("cores", snapshot)

    def test_grab_returns_samples_on_success_and_tolerates_blank_lines(self):
        # A stray blank line in /proc output must not discard the whole sample.
        with patch("rdma_sweep._run_remote_result", autospec=True,
                   side_effect=_remote_fake(**{
                       "/proc/stat":     RemoteResult(host="", command="", stdout="cpu  1 2 3 4 5 6 7 8\n\ncpu0 1 2 3 4 5 6 7 8\n"),
                       "/proc/meminfo":  RemoteResult(host="", command="", stdout="MemTotal: 100 kB\nMemFree: 40 kB\n"),
                       "/proc/softirqs": RemoteResult(host="", command="", stdout="          CPU0\n\nNET_RX:  12\n"),
                   })):
            snapshot = SysMonitor("good-host").grab()

        self.assertNotIn("error", snapshot)
        self.assertIn("cpu0", snapshot["cores"])
        self.assertEqual(snapshot["softirqs"]["NET_RX"], 12)

    def test_grab_skips_malformed_numeric_line_without_aborting(self):
        # A non-numeric token in /proc/stat or /proc/softirqs must skip only
        # that line — not raise out of grab() and abort the whole sweep.
        with patch("rdma_sweep._run_remote_result", autospec=True,
                   side_effect=_remote_fake(**{
                       "/proc/stat":     RemoteResult(host="", command="", stdout="cpu0 1 2 BAD 4 5 6 7 8\ncpu1 1 2 3 4 5 6 7 8\n"),
                       "/proc/meminfo":  RemoteResult(host="", command="", stdout="MemTotal: 100 kB\n"),
                       "/proc/softirqs": RemoteResult(host="", command="", stdout="NET_RX:  oops 12\n"),
                   })):
            snapshot = SysMonitor("good-host").grab()

        self.assertNotIn("error", snapshot)          # must not crash the sweep
        self.assertNotIn("cpu0", snapshot["cores"])  # malformed core skipped
        self.assertIn("cpu1", snapshot["cores"])     # good core retained
        self.assertNotIn("NET_RX", snapshot["softirqs"])  # malformed softirq skipped

    def test_grab_tolerates_malformed_meminfo_lines(self):
        # A "Key:" line with no value (IndexError) or a non-numeric value
        # (ValueError) must skip only that line, not abort the sweep.
        with patch("rdma_sweep._run_remote_result", autospec=True,
                   side_effect=_remote_fake(**{
                       "/proc/stat":     RemoteResult(host="", command="", stdout="cpu0 1 2 3 4 5 6 7 8\n"),
                       "/proc/meminfo":  RemoteResult(host="", command="", stdout="MemTotal:\nBadVal: notanumber kB\n  Indented: 7 kB\nMemFree: 4242 kB\n"),
                       "/proc/softirqs": RemoteResult(host="", command="", stdout="NET_RX:  12\n"),
                   })):
            snapshot = SysMonitor("good-host").grab()

        self.assertNotIn("error", snapshot)               # empty value must not IndexError-abort
        self.assertNotIn("MemTotal", snapshot["mem_kB"])  # "Key:" w/ no value skipped
        self.assertNotIn("BadVal", snapshot["mem_kB"])    # non-numeric value skipped
        self.assertEqual(snapshot["mem_kB"]["MemFree"], 4242)  # good line retained
        self.assertEqual(snapshot["mem_kB"]["Indented"], 7)    # key must be .strip()ed

    def test_grab_reports_error_when_read_succeeds_but_is_unparseable(self):
        # SSH succeeds (rc 0) but /proc/stat content has no usable cpu lines
        # (truncated read, banner/error contamination, wrong format).  The
        # per-line guards silently drop everything, leaving empty cores with NO
        # "error" key — which run_sweep's *_sys_*_ok flags would then report as
        # a clean, genuinely-idle host, and the CSV would render as blank/0 cpu.
        # grab() must flag this so it can never masquerade as a valid data point.
        with patch("rdma_sweep._run_remote_result", autospec=True,
                   side_effect=_remote_fake(**{
                       "/proc/stat":     RemoteResult(host="", command="", stdout="Connection reset by peer\n"),
                       "/proc/meminfo":  RemoteResult(host="", command="", stdout="MemTotal: 100 kB\n"),
                       "/proc/softirqs": RemoteResult(host="", command="", stdout="NET_RX: 1\n"),
                   })):
            snapshot = SysMonitor("reachable-but-garbage").grab()

        self.assertIn("error", snapshot)        # not a falsely-ok empty snapshot
        self.assertNotIn("cores", snapshot)     # no usable data returned at all

    def test_grab_reports_error_when_meminfo_unparseable(self):
        # The memory arm of the same guard: /proc/stat parses fine but
        # /proc/meminfo (rc 0) yields no usable keys, so mem is empty.  A blank
        # mem column reads as "no allocation"; flag it instead of masquerading.
        with patch("rdma_sweep._run_remote_result", autospec=True,
                   side_effect=_remote_fake(**{
                       "/proc/stat":     RemoteResult(host="", command="", stdout="cpu0 1 2 3 4 5 6 7 8\n"),
                       "/proc/meminfo":  RemoteResult(host="", command="", stdout="garbage with no colon\n"),
                       "/proc/softirqs": RemoteResult(host="", command="", stdout="NET_RX: 1\n"),
                   })):
            snapshot = SysMonitor("reachable-but-garbage-mem").grab()

        self.assertIn("error", snapshot)
        self.assertNotIn("mem_kB", snapshot)


# ---------------------------------------------------------------------------
# _parse_perf_line  (10 scenarios)
# ---------------------------------------------------------------------------

class TestParsePerfLine(unittest.TestCase):
    """_parse_perf_line parses one perf report line into (key, self_pct)."""

    def test_normal_line(self):
        key, pct = _parse_perf_line("    10.00%     5.00%  ib_write_bw  libc.so  [.] read\n")
        self.assertEqual(key, "read@libc.so")
        self.assertEqual(pct, 5.0)

    def test_kernel_symbol(self):
        key, pct = _parse_perf_line("     8.00%     8.00%  swapper  [kernel.kallsyms]  [k] do_idle\n")
        self.assertEqual(key, "do_idle")
        self.assertEqual(pct, 8.0)

    def test_demangled_cpp_symbol(self):
        key, pct = _parse_perf_line("    10.00%    10.00%  ib_write_bw  libstdc++.so  [.] operator new(unsigned long)\n")
        self.assertEqual(key, "operator new(unsigned long)@libstdc++.so")
        self.assertEqual(pct, 10.0)

    def test_different_dso_disambiguated(self):
        key, pct = _parse_perf_line("     8.00%     8.00%  ib_write_bw  liba.so  [.] poll\n")
        self.assertEqual(key, "poll@liba.so")
        self.assertEqual(pct, 8.0)

    def test_comment_line_returns_none(self):
        self.assertIsNone(_parse_perf_line("# To display the perf.data header info\n"))

    def test_empty_line_returns_none(self):
        self.assertIsNone(_parse_perf_line(""))
        self.assertIsNone(_parse_perf_line("   \n"))

    def test_line_without_annotation_returns_none(self):
        self.assertIsNone(_parse_perf_line("    10.00%     5.00%  ib_write_bw  libc.so  read\n"))

    def test_too_few_parts_returns_none(self):
        self.assertIsNone(_parse_perf_line("    10.00%     5.00%  ib_write_bw\n"))

    def test_non_numeric_self_pct_returns_none(self):
        self.assertIsNone(_parse_perf_line("    10.00%     BAD%  ib_write_bw  libc.so  [.] read\n"))

    def test_zero_self_pct_returns_none(self):
        self.assertIsNone(_parse_perf_line("    10.00%     0.00%  ib_write_bw  libc.so  [.] read\n"))


class TestParsePerfReportDisambiguation(unittest.TestCase):
    """Symbol disambiguation cases for _parse_perf_report."""

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


# ---------------------------------------------------------------------------
# _env_prefix  (5 scenarios)
# ---------------------------------------------------------------------------

class TestEnvPrefix(unittest.TestCase):
    """_env_prefix builds the env-var prefix string for perftest commands.

    The rdma_core_lib → LD_LIBRARY_PATH injection and invalid-key guard are
    never asserted directly; they run only transitively through run_perftest
    fakes that don't inspect the emitted prefix.
    """

    def test_empty_config_produces_empty_prefix(self):
        from rdma_sweep import _env_prefix
        self.assertEqual(_env_prefix({"env": {}}), "")

    def test_rdma_core_lib_injects_ld_library_path(self):
        from rdma_sweep import _env_prefix
        env = _env_prefix({"rdma_core_lib": "/opt/rdma/lib", "env": {}})
        self.assertIn("LD_LIBRARY_PATH=/opt/rdma/lib", env)

    def test_explicit_ld_library_path_not_overridden(self):
        from rdma_sweep import _env_prefix
        env = _env_prefix({
            "rdma_core_lib": "/opt/rdma/lib",
            "env": {"LD_LIBRARY_PATH": "/custom"},
        })
        self.assertIn("LD_LIBRARY_PATH=/custom", env)
        self.assertNotIn("/opt/rdma/lib", env)

    def test_invalid_var_name_raises(self):
        from rdma_sweep import _env_prefix
        with self.assertRaises(ValueError):
            _env_prefix({"env": {"2BAD": "value"}})

    def test_none_values_filtered_out(self):
        from rdma_sweep import _env_prefix
        env = _env_prefix({"env": {"FOO": "bar", "BAZ": None}})
        self.assertIn("FOO=bar", env)
        self.assertNotIn("BAZ", env)


# ---------------------------------------------------------------------------
# _filtered_perftest_args  (4 scenarios)
# ---------------------------------------------------------------------------

class TestFilteredPerftestArgs(unittest.TestCase):
    """_filtered_perftest_args strips output-formatting flags the tool adds itself.

    All run_perftest tests pass extra_args with no --out_json* tokens, so the
    entire filtering body (lines 317-327) never executed in the suite.
    """

    def test_nothing_to_filter(self):
        from rdma_sweep import _filtered_perftest_args
        self.assertEqual(_filtered_perftest_args(["-s", "64K"]), ["-s", "64K"])

    def test_strips_out_json_flags(self):
        from rdma_sweep import _filtered_perftest_args
        self.assertEqual(
            _filtered_perftest_args(["-s", "64K", "--out_json", "--out-json"]),
            ["-s", "64K"],
        )

    def test_strips_out_json_file_with_value(self):
        from rdma_sweep import _filtered_perftest_args
        self.assertEqual(
            _filtered_perftest_args(
                ["-s", "64K", "--out_json_file", "out.json", "-D", "5"],
            ),
            ["-s", "64K"],
        )

    def test_strips_attached_forms(self):
        from rdma_sweep import _filtered_perftest_args
        self.assertEqual(
            _filtered_perftest_args(
                ["-s", "64K", "--out-json-file=x.json", "-D5"],
            ),
            ["-s", "64K"],
        )


# ---------------------------------------------------------------------------
# _has_flag  (3 scenarios)
# ---------------------------------------------------------------------------

class TestHasFlag(unittest.TestCase):
    """_has_flag detects whether a flag is present in an arg list.

    The --iters=100 attached-value form was never exercised, only bare -n.
    """

    def test_bare_flag(self):
        from rdma_sweep import _has_flag
        self.assertTrue(_has_flag(["-n", "100"], "-n", "--iters"))

    def test_attached_value_form(self):
        from rdma_sweep import _has_flag
        self.assertTrue(_has_flag(["--iters=100"], "-n", "--iters"))

    def test_short_form_attached_value(self):
        from rdma_sweep import _has_flag
        self.assertTrue(_has_flag(["-n1000"], "-n", "--iters"))

    def test_absent_flag(self):
        from rdma_sweep import _has_flag
        self.assertFalse(_has_flag(["-s", "64K"], "-n", "--iters"))

    def test_short_flag_no_false_positive_on_prefix_match(self):
        """``-no_cma`` must NOT match ``-n`` — regression guard."""
        from rdma_sweep import _has_flag
        self.assertFalse(_has_flag(["-no_cma"], "-n", "--iters"))
        self.assertFalse(_has_flag(["--no_cma"], "-n", "--iters"))


# ---------------------------------------------------------------------------
# _build_perftest_cmdline  (3 scenarios)
# ---------------------------------------------------------------------------

class TestBuildPerftestCmdline(unittest.TestCase):
    """_build_perftest_cmdline builds args_str and duration_arg."""

    def test_no_iter_flag_uses_duration(self):
        args_str, duration_arg = _build_perftest_cmdline(["-s", "64K"], 10)
        self.assertEqual(args_str, "-s 64K")
        self.assertEqual(duration_arg, "-D 10")

    def test_bare_n_flag_omits_duration(self):
        args_str, duration_arg = _build_perftest_cmdline(["-n", "1000"], 10)
        self.assertEqual(duration_arg, "")

    def test_iters_flag_omits_duration(self):
        args_str, duration_arg = _build_perftest_cmdline(["--iters=500"], 10)
        self.assertEqual(duration_arg, "")


# ---------------------------------------------------------------------------
# format_size / parse_size  (6 scenarios)
# ---------------------------------------------------------------------------

class TestFormatParseSize(unittest.TestCase):
    """format_size and parse_size round-trip for memory delta reporting.

    The negative branch of format_size is real (memory can be freed), but was
    never tested directly.
    """

    def test_format_bytes(self):
        from rdma_sweep import format_size
        self.assertEqual(format_size(0), "0B")
        self.assertEqual(format_size(512), "512B")

    def test_format_kb(self):
        from rdma_sweep import format_size
        self.assertEqual(format_size(1536), "1.5K")

    def test_format_mb(self):
        from rdma_sweep import format_size
        self.assertEqual(format_size(1048576), "1.0M")

    def test_format_negative(self):
        from rdma_sweep import format_size
        self.assertEqual(format_size(-1048576), "-1.0M")

    def test_format_tb(self):
        from rdma_sweep import format_size
        self.assertEqual(format_size(1024**4), "1.0T")
        self.assertEqual(format_size(int(1.5 * 1024**4)), "1.5T")

    def test_parse_roundtrip(self):
        from rdma_sweep import parse_size
        self.assertEqual(parse_size("64K"), 65536)
        self.assertEqual(parse_size("2G"), 2 * 1024**3)
        self.assertEqual(parse_size("100"), 100)

    def test_parse_negative(self):
        from rdma_sweep import parse_size
        self.assertEqual(parse_size("-1.0M"), -1048576)


class TestLog2Px(unittest.TestCase):
    """_log2_px computes log2-scaled x-positions for SVG charts."""

    def test_basic_log2_scaling(self):
        from rdma_sweep import _log2_px
        # xmn=1, xmx=8 → log2 range = [0, 3]
        # v=4 → log2(4)=2, fraction = 2/3 → px = cx + 2/3 * cw
        result = _log2_px(v=4, cx=10.0, xmn=1.0, xmx=8.0, cw=90.0)
        self.assertAlmostEqual(result, 10 + (2/3) * 90)

    def test_single_point_no_scale(self):
        from rdma_sweep import _log2_px
        # xmn == xmx → no scaling possible → cx + cw/2
        result = _log2_px(v=4, cx=10.0, xmn=4.0, xmx=4.0, cw=100.0)
        self.assertEqual(result, 60.0)

    def test_min_value_left_edge(self):
        from rdma_sweep import _log2_px
        # v == xmn → log2 fraction = 0 → px = cx
        result = _log2_px(v=1, cx=10.0, xmn=1.0, xmx=8.0, cw=90.0)
        self.assertAlmostEqual(result, 10.0)

    def test_max_value_right_edge(self):
        from rdma_sweep import _log2_px
        # v == xmx → log2 fraction = 1 → px = cx + cw
        result = _log2_px(v=8, cx=10.0, xmn=1.0, xmx=8.0, cw=90.0)
        self.assertAlmostEqual(result, 100.0)


# ---------------------------------------------------------------------------
# _as_float edge cases  (3 scenarios)
# ---------------------------------------------------------------------------

class TestAsFloat(unittest.TestCase):
    """_as_float conversion edge cases for chart data."""

    def test_empty_string_returns_default(self):
        from rdma_sweep import _as_float
        self.assertEqual(_as_float("", default=0.0), 0.0)

    def test_none_value_returns_default(self):
        from rdma_sweep import _as_float
        self.assertEqual(_as_float(None, default=0.0), 0.0)

    def test_non_parseable_returns_default(self):
        from rdma_sweep import _as_float
        self.assertEqual(_as_float("not-a-number", default=42.0), 42.0)


# ---------------------------------------------------------------------------
# SysMonitor.extract_mem / compute_mem_delta  (5 scenarios)
# ---------------------------------------------------------------------------

class TestSysMonitorMemHelpers(unittest.TestCase):
    """SysMonitor.extract_mem and compute_mem_delta produce correct values.

    Both run transitively through _idle_meta tests with pre-formatted strings,
    but the producing functions (which operate on raw mem_kB dicts) were never
    pinned directly.
    """

    def test_extract_mem_computes_used(self):
        after = {"mem_kB": {"MemTotal": 1048576, "MemFree": 524288, "Buffers": 0, "Cached": 0}}
        m = SysMonitor.extract_mem(after)
        self.assertEqual(m["MemUsed"], "512.0M")
        self.assertEqual(m["MemTotal"], "1.0G")

    def test_extract_mem_empty_input(self):
        self.assertEqual(SysMonitor.extract_mem({}), {})

    def test_compute_mem_delta_positive(self):
        before = {"mem_kB": {"MemTotal": 1048576, "MemFree": 524288, "Buffers": 0, "Cached": 0}}
        after = {"mem_kB": {"MemTotal": 1048576, "MemFree": 262144, "Buffers": 0, "Cached": 0}}
        d = SysMonitor.compute_mem_delta(before, after)
        self.assertIn("256.0M", d["MemUsedDelta"])

    def test_compute_mem_delta_negative(self):
        before = {"mem_kB": {"MemTotal": 1048576, "MemFree": 262144, "Buffers": 0, "Cached": 0}}
        after = {"mem_kB": {"MemTotal": 1048576, "MemFree": 524288, "Buffers": 0, "Cached": 0}}
        d = SysMonitor.compute_mem_delta(before, after)
        self.assertIn("-256.0M", d["MemUsedDelta"])

    def test_compute_mem_delta_empty_before(self):
        self.assertEqual(
            SysMonitor.compute_mem_delta({}, {"mem_kB": {"MemTotal": 100}}),
            {},
        )


# ---------------------------------------------------------------------------
# RemoteResult.error_summary remaining branches  (3 scenarios)
# ---------------------------------------------------------------------------

class TestErrorSummary(unittest.TestCase):
    """RemoteResult.error_summary() covers timed_out / exception / exit-N.

    TestRemoteResult covered success + stderr-on-failure, but the three other
    branches (timed_out message, exception text, exit-N fallback) were missed.
    """

    def test_timed_out_returns_timeout_message(self):
        r = RemoteResult(host="h", command="sleep 10", timed_out=True)
        self.assertIn("timed out", r.error_summary())

    def test_exception_returns_exception_text(self):
        r = RemoteResult(host="h", command="cmd", exception="boom")
        self.assertEqual(r.error_summary(), "boom")

    def test_failure_no_stderr_returns_exit_code(self):
        r = RemoteResult(host="h", command="false", returncode=1, stderr="")
        self.assertEqual(r.error_summary(), "exit 1")


# ---------------------------------------------------------------------------
# /usr/bin/time client-usage parsing  (1 scenario)
# ---------------------------------------------------------------------------

class TestClientUsageParsing(unittest.TestCase):
    """The /usr/bin/time output fields are parsed into client_usage.

    Multiple run_perftest tests feed a realistic time_fmt line into the fake,
    but zero assertions checked result["_process"]["client_usage"].
    """

    def test_client_usage_parsed_from_time_output(self):
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

        fake_run_remote_result = _remote_fake(**{
            "cat /tmp/test_server.pid": RemoteResult(host="", command="", stdout="123\n"),
            "cat /tmp/test_time.out":   RemoteResult(host="", command="", stdout="2.5 1.3 75% 2048 0 0\n"),
            "cat /tmp/test_out.json":   RemoteResult(host="", command="", stdout='{"results": {"BW_average": 42, "MsgRate": 0.1}}'),
        })

        with (
            patch("rdma_sweep._run_remote_result", autospec=True, side_effect=fake_run_remote_result),
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

        usage = result["_process"]["client_usage"]
        self.assertEqual(usage["client_user_sec"], "2.5")
        self.assertEqual(usage["client_sys_sec"], "1.3")
        self.assertEqual(usage["client_cpu_pct"], "75")
        self.assertEqual(usage["client_max_rss_kb"], "2048")


# ---------------------------------------------------------------------------
# rdma_config direct unit tests  (11 scenarios)
# ---------------------------------------------------------------------------

class TestRdmaConfigUnits(unittest.TestCase):
    """Direct unit tests for rdma_config functions.

    parse_bool, strip_user, is_loopback, endpoint_config,
    resolve_perftest_paths, and the legacy flat schema were tested only
    end-to-end through _runtime_config — several branches never executed.
    """

    def test_parse_bool_int_zero_false(self):
        from rdma_config import parse_bool
        self.assertFalse(parse_bool(0, "k"))

    def test_parse_bool_int_one_true(self):
        from rdma_config import parse_bool
        self.assertTrue(parse_bool(1, "k"))

    def test_parse_bool_string_true_branches(self):
        from rdma_config import parse_bool
        for v in ("yes", "true", "on", "1"):
            self.assertTrue(parse_bool(v, "k"), f"parse_bool({v!r}) should be True")

    def test_parse_bool_string_false_branches(self):
        from rdma_config import parse_bool
        for v in ("no", "false", "off", "0"):
            self.assertFalse(parse_bool(v, "k"), f"parse_bool({v!r}) should be False")

    def test_strip_user(self):
        from rdma_config import strip_user
        self.assertEqual(strip_user("user@host"), "host")
        self.assertEqual(strip_user("host"), "host")

    def test_is_loopback_names(self):
        from rdma_config import is_loopback
        self.assertTrue(is_loopback("localhost"))
        self.assertTrue(is_loopback("ip6-localhost"))
        self.assertTrue(is_loopback("::1"))

    def test_endpoint_config_string_form(self):
        from rdma_config import endpoint_config
        ep = endpoint_config({"server": "srv-host"}, "server")
        self.assertEqual(ep["host"], "srv-host")

    def test_endpoint_config_rdma_address_alias(self):
        from rdma_config import endpoint_config
        # rdma_address → address alias only fires when "address" is absent.
        # For "server", setdefault("address", ...) runs first with the top-level
        # server_address/server_addr keys.  Use "client" (which has no setdefault
        # for address) to exercise the alias cleanly.
        ep = endpoint_config(
            {"client": {"host": "h", "rdma_address": "10.0.0.2"}},
            "client",
        )
        self.assertEqual(ep["address"], "10.0.0.2")

    def test_endpoint_config_invalid_type_raises(self):
        """endpoint_config with non-str/non-dict/non-None type raises ValueError."""
        from rdma_config import endpoint_config
        with self.assertRaises(ValueError):
            endpoint_config({"server": 42}, "server")

    def test_resolve_perftest_paths_substitutes_tokens(self):
        from rdma_config import resolve_perftest_paths, DEFAULT_PERFTEST_CONFIG
        cfg = dict(DEFAULT_PERFTEST_CONFIG)
        cfg["tmp_dir"] = "/tmp/sweep_{run_id}"
        resolved = resolve_perftest_paths(cfg, run_id="abc123")
        self.assertEqual(resolved["tmp_dir"], "/tmp/sweep_abc123")
        self.assertEqual(resolved["json_file"], "/tmp/sweep_abc123/perftest_out.json")

    def test_legacy_flat_config_accepted(self):
        from rdma_config import runtime_config
        runtime = runtime_config({
            "server_host": "server-ssh",
            "server_address": "10.0.0.2",
            "client_host": "client-ssh",
            "perftest_dir": "/opt/perftest",
        })
        self.assertEqual(runtime["server"]["host"], "server-ssh")
        self.assertEqual(runtime["perftest"]["dir"], "/opt/perftest")

    def test_legacy_rdma_core_lib_promoted(self):
        from rdma_config import runtime_config
        runtime = runtime_config({
            "server_host": "server-ssh",
            "server_address": "10.0.0.2",
            "client_host": "client-ssh",
            "perftest_dir": "/opt/perftest",
            "rdma_core_lib": "/opt/rdma/lib",
        })
        self.assertEqual(runtime["perftest"]["rdma_core_lib"], "/opt/rdma/lib")

    def test_ssh_options_none_yields_empty_options(self):
        """ssh.options=None (YAML null) produces empty list, not crash."""
        from rdma_config import runtime_config
        runtime = runtime_config({
            "server": {"host": "srv", "address": "10.0.0.1"},
            "client": {"host": "cli"},
            "perftest": {"dir": "/usr/bin"},
            "ssh": {"options": None},
        })
        self.assertEqual(runtime["ssh"]["options"], [])

    def test_ssh_options_empty_stays_empty(self):
        """ssh.options=[] keeps empty list (explicit user intent)."""
        from rdma_config import runtime_config
        runtime = runtime_config({
            "server": {"host": "srv", "address": "10.0.0.1"},
            "client": {"host": "cli"},
            "perftest": {"dir": "/usr/bin"},
            "ssh": {"options": []},
        })
        self.assertEqual(runtime["ssh"]["options"], [])

    def test_runtime_config_non_mapping_perftest_env_raises(self):
        """perftest.env that is not a dict raises ValueError."""
        from rdma_config import runtime_config
        # host/address guards all pass; ValueError must come from the env check
        with self.assertRaisesRegex(ValueError, "perftest.env must be a mapping"):
            runtime_config({
                "server": {"host": "srv", "address": "10.0.0.1"},
                "client": {"host": "cli"},
                "perftest": {"dir": "/usr/bin", "env": "not-a-dict"},
            })

    def test_runtime_config_missing_server_host_raises(self):
        """Missing or empty server.host without legacy fallback raises ValueError."""
        from rdma_config import runtime_config
        with self.assertRaisesRegex(ValueError, "config must set server.host"):
            runtime_config({
                "server": {"address": "10.0.0.1"},  # no "host" key at all
                "client": {"host": "cli"},
                "perftest": {"dir": "/usr/bin"},
            })

    def test_runtime_config_null_server_address_raises(self):
        """server.address explicitly set to YAML null raises ValueError.

        The .get(key, default) pattern returns None when the key exists with a
        null value — str(None) == "None" would slip past the empty-string check
        without the ``or ""`` fallback.
        """
        from rdma_config import runtime_config
        with self.assertRaisesRegex(ValueError, "config must set server.address"):
            runtime_config({
                "server": {"host": "srv", "address": None},
                "client": {"host": "cli"},
                "perftest": {"dir": "/usr/bin"},
            })

    def test_runtime_config_null_test_defaults_to_ib_write_bw(self):
        """test: null falls back to ib_write_bw instead of str(None) == 'None'."""
        from rdma_config import runtime_config
        runtime = runtime_config({
            "server": {"host": "srv", "address": "10.0.0.1"},
            "client": {"host": "cli"},
            "perftest": {"dir": "/usr/bin"},
            "test": None,
        })
        self.assertEqual(runtime["test"], "ib_write_bw")

    def test_runtime_config_null_tmp_dir_falls_back_to_default(self):
        """perftest.tmp_dir: null falls back to default template."""
        from rdma_config import resolve_perftest_paths, DEFAULT_PERFTEST_CONFIG
        from copy import deepcopy
        cfg = deepcopy(DEFAULT_PERFTEST_CONFIG)
        cfg["tmp_dir"] = None  # simulate YAML perftest: {tmp_dir: null}
        resolved = resolve_perftest_paths(cfg, run_id="test123")
        self.assertIn("test123", resolved["tmp_dir"])
        self.assertNotIn("None", resolved["tmp_dir"])

    def test_runtime_config_null_perftest_dir_raises(self):
        """perftest.dir: null raises ValueError instead of using 'None' as path."""
        from rdma_config import runtime_config
        with self.assertRaisesRegex(ValueError, "config must set perftest.dir"):
            runtime_config({
                "server": {"host": "srv", "address": "10.0.0.1"},
                "client": {"host": "cli"},
                "perftest": {"dir": None},
            })


# ---------------------------------------------------------------------------
# run_local_result success and sudo paths  (3 scenarios)
# ---------------------------------------------------------------------------

class TestRunLocalResult(unittest.TestCase):
    """run_local_result success path and sudo argv construction.

    Only the timeout path was tested; the happy path, sudo prefix, and
    generic-exception branch were all missed.
    """

    def test_success_propagates_output(self):
        completed = subprocess.CompletedProcess(
            args=["bash", "-c", "echo ok"],
            returncode=0, stdout="ok\n", stderr="",
        )
        with patch("rdma_remote.subprocess.run", return_value=completed):
            result = run_local_result("echo ok", sudo=False)
        self.assertTrue(result.ok)
        self.assertEqual(result.stdout, "ok\n")

    def test_sudo_true_prepends_sudo(self):
        with patch("rdma_remote.subprocess.run") as run:
            run.return_value = subprocess.CompletedProcess(args=[], returncode=0)
            run_local_result("echo ok", sudo=True)
        self.assertEqual(run.call_args.args[0][0], "sudo")

    def test_generic_exception_returns_error(self):
        with patch("rdma_remote.subprocess.run", side_effect=RuntimeError("spawn failed")):
            result = run_local_result("echo ok", sudo=False)
        self.assertFalse(result.ok)
        self.assertEqual(result.exception, "spawn failed")


# ---------------------------------------------------------------------------
# init_local_hosts  (2 scenarios)
# ---------------------------------------------------------------------------

class TestInitLocalHosts(unittest.TestCase):
    """init_local_hosts populates _LOCAL_HOSTS from hostname commands.

    The function runs at import, but the hostname parsing and exception
    swallowing were never tested with controlled input.
    """

    def test_baseline_always_present(self):
        from rdma_remote import init_local_hosts, _LOCAL_HOSTS
        init_local_hosts()
        self.assertIn("127.0.0.1", _LOCAL_HOSTS)
        self.assertIn("localhost", _LOCAL_HOSTS)

    def test_hostname_output_added(self):
        from rdma_remote import init_local_hosts, _LOCAL_HOSTS
        with patch("rdma_remote.subprocess.check_output", return_value="myhost myhost-alias 10.0.0.5"):
            init_local_hosts()
        self.assertIn("myhost", _LOCAL_HOSTS)
        self.assertIn("myhost-alias", _LOCAL_HOSTS)
        self.assertIn("10.0.0.5", _LOCAL_HOSTS)


# ---------------------------------------------------------------------------
# sweep_config  (5 scenarios)
# ---------------------------------------------------------------------------

class TestSweepConfig(unittest.TestCase):
    """sweep_config yields every parameter combination.

    sweep_config is only transitively exercised through run_sweep (which itself
    is never called in the suite).  The expansion + cartesian-product logic
    never ran in the test suite, so a regression in itertools.product could
    silently skip combos or yield duplicates while every test stays green.
    """

    def test_single_param_expands(self):
        from rdma_sweep import sweep_config
        combos = list(sweep_config({
            "sweep": [{"name": "qp", "values": [1, 4, 16]}],
            "fixed": {"port": 18515},
        }))
        self.assertEqual(len(combos), 3)
        qps = [c["qp"] for c in combos]
        self.assertEqual(qps, [1, 4, 16])
        # fixed params applied to every combo
        self.assertTrue(all(c["port"] == 18515 for c in combos))

    def test_two_params_cartesian_product(self):
        from rdma_sweep import sweep_config
        combos = list(sweep_config({
            "sweep": [
                {"name": "msg_size", "values": [64, 1024]},
                {"name": "qp", "values": [1, 4]},
            ],
            "fixed": {},
        }))
        self.assertEqual(len(combos), 4)
        pairs = [(c["msg_size"], c["qp"]) for c in combos]
        self.assertIn((64, 1), pairs)
        self.assertIn((64, 4), pairs)
        self.assertIn((1024, 1), pairs)
        self.assertIn((1024, 4), pairs)

    def test_range_param_expands(self):
        from rdma_sweep import sweep_config
        combos = list(sweep_config({
            "sweep": [{"name": "qp", "from": 1, "to": 3, "step": 1}],
        }))
        self.assertEqual(len(combos), 3)
        self.assertEqual([c["qp"] for c in combos], [1, 2, 3])

    def test_empty_sweep_yields_no_combos(self):
        from rdma_sweep import sweep_config
        combos = list(sweep_config({"sweep": []}))
        # itertools.product(*[]) yields one empty tuple, so an empty sweep
        # spec produces exactly one combo containing only fixed params.
        self.assertEqual(combos, [{}])

    def test_fixed_merged_with_every_combo(self):
        from rdma_sweep import sweep_config
        combos = list(sweep_config({
            "sweep": [{"name": "qp", "values": [1]}],
            "fixed": {"port": 18515, "device": "roce0"},
        }))
        self.assertEqual(len(combos), 1)
        self.assertEqual(combos[0]["port"], 18515)
        self.assertEqual(combos[0]["device"], "roce0")
        self.assertEqual(combos[0]["qp"], 1)

    def test_sweep_entry_missing_name_raises_value_error(self):
        """sweep entry without 'name' raises ValueError (not KeyError)."""
        from rdma_sweep import sweep_config
        with self.assertRaises(ValueError) as ctx:
            list(sweep_config({"sweep": [{}]}))
        self.assertIn("name", str(ctx.exception))

    def test_sweep_entry_non_dict_raises_value_error(self):
        """sweep entry that is not a dict raises ValueError."""
        from rdma_sweep import sweep_config
        with self.assertRaises(ValueError) as ctx:
            list(sweep_config({"sweep": ["not a dict"]}))
        self.assertIn("name", str(ctx.exception))

    def test_sweep_null_is_same_as_empty(self):
        """sweep: null (YAML) yields one empty combo, no crash."""
        from rdma_sweep import sweep_config
        combos = list(sweep_config({"sweep": None}))
        self.assertEqual(combos, [{}])

    def test_fixed_null_skips_fixed_params(self):
        """fixed: null (YAML) does not crash; yields raw sweep combos."""
        from rdma_sweep import sweep_config
        combos = list(sweep_config({
            "sweep": [{"name": "qp", "values": [1, 2]}],
            "fixed": None,
        }))
        self.assertEqual(len(combos), 2)
        self.assertEqual(combos[0], {"qp": 1})
        self.assertEqual(combos[1], {"qp": 2})

    def test_qp_zero_raises_value_error(self):
        """qp=0 raises ValueError with descriptive message."""
        from rdma_sweep import sweep_config
        with self.assertRaises(ValueError) as ctx:
            list(sweep_config({"sweep": [{"name": "qp", "values": [0, 1]}]}))
        self.assertIn("QP must be positive", str(ctx.exception))

    def test_qp_negative_raises_value_error(self):
        """qp<0 raises the same positive-QP ValueError (float coercion path)."""
        from rdma_sweep import sweep_config
        with self.assertRaises(ValueError) as ctx:
            list(sweep_config({"sweep": [{"name": "qp", "values": [-2, 1]}]}))
        self.assertIn("QP must be positive", str(ctx.exception))

    def test_qp_numeric_string_is_accepted(self):
        """A quoted-but-numeric qp (e.g. YAML '2') coerces and passes.

        sweep_config used to compare ``qp <= 0`` directly; a string value
        raised an opaque TypeError that aborted the whole sweep.  Numeric
        strings must now coerce cleanly so a stray quote is forgiving.
        """
        from rdma_sweep import sweep_config
        combos = list(sweep_config({"sweep": [{"name": "qp", "values": ["1", "2"]}]}))
        self.assertEqual([c["qp"] for c in combos], ["1", "2"])

    def test_qp_non_numeric_string_raises_clear_error(self):
        """A non-numeric qp raises a clear ValueError, not an opaque TypeError."""
        from rdma_sweep import sweep_config
        with self.assertRaises(ValueError) as ctx:
            list(sweep_config({"sweep": [{"name": "qp", "values": ["fast"]}]}))
        self.assertIn("QP must be numeric", str(ctx.exception))

    def test_qp_nan_raises_finite_error(self):
        """qp=nan (YAML .nan) is rejected, not dispatched as -q nan.

        ``float('nan') <= 0`` is False, so without an explicit finite check a
        NaN qp would pass validation and reach perftest as an always-ERR run —
        exactly what this validator exists to catch early.
        """
        from rdma_sweep import sweep_config
        with self.assertRaises(ValueError) as ctx:
            list(sweep_config({"sweep": [{"name": "qp", "values": [float("nan")]}]}))
        self.assertIn("finite", str(ctx.exception))

    def test_qp_inf_raises_finite_error(self):
        """qp=inf (YAML .inf) is rejected for the same reason as nan."""
        from rdma_sweep import sweep_config
        with self.assertRaises(ValueError) as ctx:
            list(sweep_config({"sweep": [{"name": "qp", "values": [float("inf")]}]}))
        self.assertIn("finite", str(ctx.exception))


# ---------------------------------------------------------------------------
# run_sweep  (3 scenarios)
# ---------------------------------------------------------------------------

class TestRunSweep(unittest.TestCase):
    """run_sweep orchestrates the sweep: combos → perftest → summary.

    run_sweep was never called in the suite — only its sub-components
    (run_perftest, _summary_entry, _write_csv) were tested in isolation.
    The orchestration logic (combo iteration, sys-monitor before/after,
    result writing, summary generation) never executed.
    """

    def _fake_runtime(self):
        """Return a runtime_config dict safe for JSON serialization."""
        return {
            "test": "ib_write_bw",
            "duration": 1,
            "use_gpu": False,
            "server": {"host": "server-ssh", "address": "10.0.0.2"},
            "client": {"host": "client-ssh"},
            "perftest": {"dir": "/opt/perftest", "perf_record": True, "wait_timeout": 30,
                         "default_port": 18515, "env": {}},
            "ssh": {"sudo": True, "allow_local": False, "connect_timeout": 10, "options": []},
            "report": {"title": "RDMA Perftest Sweep", "subtitle": ""},
        }

    def _clean_grab(self):
        """Return a clean SysMonitor.grab() snapshot dict."""
        return {
            "time": "2026-01-01",
            "cores": {"cpu0": {"user": 0, "nice": 0, "system": 0, "idle": 1000, "iowait": 0}},
            "mem_kB": {"MemTotal": 1048576, "MemFree": 524288, "Buffers": 0, "Cached": 0},
            "softirqs": {"NET_RX": 100},
        }

    def test_runs_combos_and_writes_results(self):
        from rdma_sweep import SysMonitor as RealSysMonitor, run_sweep
        config = {
            "test": "ib_write_bw",
            "server": {"host": "server-ssh", "address": "10.0.0.2"},
            "client": {"host": "client-ssh"},
            "perftest": {"dir": "/opt/perftest"},
            "sweep": [{"name": "qp", "values": [1, 4]}],
            "fixed": {"msg_size": 64},
            "duration": 1,
        }

        combo_count = [0]

        def fake_run_perftest(**kw):
            combo_count[0] += 1
            return {
                "results": {"BW_average": 100 * combo_count[0], "MsgRate": 1.0},
                "_process": {"server_perf": {}, "client_usage": {}},
            }

        with tempfile.TemporaryDirectory() as tmp:
            with (
                patch("rdma_sweep.run_perftest", side_effect=fake_run_perftest),
                patch("rdma_sweep._runtime_config", return_value=self._fake_runtime()),
                patch.object(RealSysMonitor, "grab", return_value=self._clean_grab()),
            ):
                files = run_sweep(config, output_dir=tmp)

            self.assertEqual(len(files), 2)  # two combos → two result files
            self.assertEqual(combo_count[0], 2)  # run_perftest called twice

            # summary.json should exist with 2 entries
            summary = json.loads((Path(tmp) / "summary.json").read_text())
            self.assertEqual(len(summary), 2)
            self.assertEqual(summary[0]["BW_average"], 100)
            self.assertEqual(summary[1]["BW_average"], 200)

            # summary.csv should exist
            self.assertTrue((Path(tmp) / "summary.csv").exists())

            # run_config.json should exist
            run_config = json.loads((Path(tmp) / "run_config.json").read_text())
            self.assertEqual(run_config["test"], "ib_write_bw")

    def test_error_propagates_to_summary(self):
        from rdma_sweep import SysMonitor as RealSysMonitor, run_sweep
        config = {
            "test": "ib_write_bw",
            "server": {"host": "server-ssh", "address": "10.0.0.2"},
            "client": {"host": "client-ssh"},
            "perftest": {"dir": "/opt/perftest"},
            "sweep": [{"name": "qp", "values": [1]}],
            "duration": 1,
        }

        def fake_run_perftest(**kw):
            return {"error": "client run failed: exit 2", "_process": {"server_perf": {}}}

        with tempfile.TemporaryDirectory() as tmp:
            with (
                patch("rdma_sweep.run_perftest", side_effect=fake_run_perftest),
                patch("rdma_sweep._runtime_config", return_value=self._fake_runtime()),
                patch.object(RealSysMonitor, "grab", return_value=self._clean_grab()),
            ):
                run_sweep(config, output_dir=tmp)

            summary = json.loads((Path(tmp) / "summary.json").read_text())
            self.assertEqual(len(summary), 1)
            self.assertTrue(summary[0]["error"])
            self.assertIn("client run failed", summary[0]["error"])

    def test_sys_ok_flags_in_meta(self):
        """Sys-monitor success flags are recorded in _meta for each result."""
        from rdma_sweep import SysMonitor as RealSysMonitor, run_sweep
        config = {
            "test": "ib_write_bw",
            "server": {"host": "server-ssh", "address": "10.0.0.2"},
            "client": {"host": "client-ssh"},
            "perftest": {"dir": "/opt/perftest"},
            "sweep": [{"name": "qp", "values": [1]}],
            "duration": 1,
        }

        def fake_run_perftest(**kw):
            return {"results": {"BW_average": 100, "MsgRate": 1.0}, "_process": {}}

        with tempfile.TemporaryDirectory() as tmp:
            with (
                patch("rdma_sweep.run_perftest", side_effect=fake_run_perftest),
                patch("rdma_sweep._runtime_config", return_value=self._fake_runtime()),
                patch.object(RealSysMonitor, "grab", return_value=self._clean_grab()),
            ):
                run_sweep(config, output_dir=tmp)

            # Check per-combo result.json has sys_ok flags
            result = json.loads((Path(tmp) / "0001" / "result.json").read_text())
            meta = result["_meta"]
            self.assertTrue(meta["server_sys_before_ok"])
            self.assertTrue(meta["server_sys_after_ok"])
            self.assertTrue(meta["client_sys_before_ok"])
            self.assertTrue(meta["client_sys_after_ok"])


# ---------------------------------------------------------------------------
# main() CLI  (4 scenarios)
# ---------------------------------------------------------------------------

class TestMainCLI(unittest.TestCase):
    """main() CLI argument parsing and mode dispatch.

    The entry point was never tested — argparse handling, YAML loading,
    and the report vs sweep mode dispatch never executed in the suite.
    """

    def test_report_mode_generates_svg(self):
        from rdma_sweep import main
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            entry = {
                "BW_average": 25000.0,
                "MsgRate": 3.8,
                "params": {"qp": 4},
            }
            (out / "summary.json").write_text(json.dumps([entry]))
            run_dir = out / "0001"
            run_dir.mkdir()
            (run_dir / "result.json").write_text(json.dumps({"_process": {"server_perf": {}}}))

            with patch("sys.argv", ["rdma_sweep", "--report", str(out)]):
                main()

            self.assertTrue((out / "chart.svg").exists())

    def test_report_mode_bool_uses_output_dir(self):
        """--report without a path falls back to --output-dir default."""
        from rdma_sweep import main
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            entry = {"BW_average": 25000.0, "MsgRate": 3.8, "params": {"qp": 4}}
            (out / "summary.json").write_text(json.dumps([entry]))
            run_dir = out / "0001"
            run_dir.mkdir()
            (run_dir / "result.json").write_text(json.dumps({"_process": {"server_perf": {}}}))

            with patch("sys.argv", ["rdma_sweep", "--report", "-o", str(out)]):
                main()

            self.assertTrue((out / "chart.svg").exists())

    def test_missing_config_prints_help_and_exits(self):
        from rdma_sweep import main
        with patch("sys.argv", ["rdma_sweep"]):
            with self.assertRaises(SystemExit) as ctx:
                main()
            self.assertEqual(ctx.exception.code, 1)

    def test_missing_yaml_prints_error_and_exits(self):
        """PyYAML not installed → error message + exit."""
        from rdma_sweep import main
        with (
            patch("sys.argv", ["rdma_sweep", "-c", "config.yaml"]),
            patch("rdma_sweep.yaml", None),
        ):
            with self.assertRaises(SystemExit) as ctx:
                main()
            self.assertNotEqual(ctx.exception.code, 0)

    def test_empty_config_file_exits_with_error(self):
        """Empty YAML file produces None config → clean error + exit."""
        from rdma_sweep import main
        fake_yaml = type("yaml", (), {"safe_load": staticmethod(lambda x: None)})()
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            f.write("")
            f.flush()
            with (
                patch("sys.argv", ["rdma_sweep", "-c", f.name]),
                patch("rdma_sweep.yaml", fake_yaml),
            ):
                with self.assertRaises(SystemExit) as ctx:
                    main()
                self.assertNotEqual(ctx.exception.code, 0)

    def test_non_dict_config_file_exits_with_error(self):
        """YAML file that parses to a non-mapping (e.g., plain scalar) → clean error + exit."""
        from rdma_sweep import main
        fake_yaml = type("yaml", (), {"safe_load": staticmethod(lambda x: "just a scalar")})()
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            f.write("just a scalar\n")
            f.flush()
            with (
                patch("sys.argv", ["rdma_sweep", "-c", f.name]),
                patch("rdma_sweep.yaml", fake_yaml),
            ):
                with self.assertRaises(SystemExit) as ctx:
                    main()
                self.assertNotEqual(ctx.exception.code, 0)

    def test_main_config_missing_sweep_key_exits(self):
        """Config without 'sweep' key prints error and exits."""
        from rdma_sweep import main
        fake_yaml = type("yaml", (), {"safe_load": staticmethod(lambda x: {"test": "ib_write_bw"})})()
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            f.write("test: ib_write_bw\n")
            f.flush()
            with (
                patch("sys.argv", ["rdma_sweep", "-c", f.name]),
                patch("rdma_sweep.yaml", fake_yaml),
            ):
                with self.assertRaises(SystemExit) as ctx:
                    main()
                self.assertNotEqual(ctx.exception.code, 0)

    def test_main_sweep_execution_calls_init_and_run_sweep(self):
        """main() with valid config calls init_local_hosts() and run_sweep()."""
        from rdma_sweep import main
        fake_yaml = type("yaml", (), {
            "safe_load": staticmethod(lambda x: {
                "sweep": [{"name": "qp", "values": [1]}],
                "test": "ib_write_bw",
                "server": {"host": "srv", "address": "10.0.0.1"},
                "client": {"host": "cli"},
                "perftest": {"dir": "/usr/bin"},
            }),
        })()
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            f.write("")
            f.flush()
            with (
                patch("sys.argv", ["rdma_sweep", "-c", f.name]),
                patch("rdma_sweep.yaml", fake_yaml),
                patch("rdma_sweep.init_local_hosts") as init_mock,
                patch("rdma_sweep.run_sweep") as run_mock,
            ):
                main()

            init_mock.assert_called_once()
            run_mock.assert_called_once()
            # Verify run_sweep received the parsed config
            args, _ = run_mock.call_args
            self.assertEqual(args[0]["test"], "ib_write_bw")

    def test_config_file_not_found_exits(self):
        """Non-existent config file raises FileNotFoundError → clean error + exit."""
        from rdma_sweep import main
        with patch("sys.argv", ["rdma_sweep", "-c", "/nonexistent/path/config.yaml"]):
            with self.assertRaises(SystemExit) as ctx:
                main()
            self.assertNotEqual(ctx.exception.code, 0)

    def test_config_yaml_syntax_error_exits(self):
        """Malformed YAML content raises yaml.YAMLError → clean error + exit.

        Uses the real ``rdma_sweep.yaml`` (not mocked) so the try/except block
        in ``main()`` catches the parser error from ``yaml.safe_load``.
        """
        from rdma_sweep import main
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "config.yaml"
            p.write_text(": broken yaml [\n")
            with patch("sys.argv", ["rdma_sweep", "-c", str(p)]):
                with self.assertRaises(SystemExit) as ctx:
                    main()
                self.assertNotEqual(ctx.exception.code, 0)

    def test_config_file_permission_error_exits(self):
        """Config file exists but is unreadable → clean error + exit."""
        from rdma_sweep import main
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "config.yaml"
            p.write_text("test: ib_write_bw\n")
            with (
                patch("sys.argv", ["rdma_sweep", "-c", str(p)]),
                patch("pathlib.Path.read_text", side_effect=PermissionError("Permission denied")),
            ):
                with self.assertRaises(SystemExit) as ctx:
                    main()
                self.assertNotEqual(ctx.exception.code, 0)

    def test_config_file_os_error_exits(self):
        """OSError from config file read → clean error + exit."""
        from rdma_sweep import main
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "config.yaml"
            p.write_text("test: ib_write_bw\n")
            with (
                patch("sys.argv", ["rdma_sweep", "-c", str(p)]),
                patch("pathlib.Path.read_text", side_effect=OSError(5, "Input/output error")),
            ):
                with self.assertRaises(SystemExit) as ctx:
                    main()
                self.assertNotEqual(ctx.exception.code, 0)

    def test_config_sweep_not_a_list_exits(self):
        """'sweep' key is not a list → clean error (type name in message) + exit.

        The temp file content is irrelevant — yaml.safe_load is mocked.
        """
        from rdma_sweep import main
        fake_yaml = type("yaml", (), {"safe_load": staticmethod(lambda x: {"sweep": "string-value"})})()
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "config.yaml"
            p.write_text("ignored\n")
            with (
                patch("sys.argv", ["rdma_sweep", "-c", str(p)]),
                patch("rdma_sweep.yaml", fake_yaml),
            ):
                with self.assertRaises(SystemExit) as ctx:
                    main()
                self.assertNotEqual(ctx.exception.code, 0)

    def test_config_sweep_empty_list_exits(self):
        """'sweep' is an empty list → clean error + exit.

        The temp file content is irrelevant — yaml.safe_load is mocked.
        """
        from rdma_sweep import main
        fake_yaml = type("yaml", (), {"safe_load": staticmethod(lambda x: {"sweep": []})})()
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "config.yaml"
            p.write_text("ignored\n")
            with (
                patch("sys.argv", ["rdma_sweep", "-c", str(p)]),
                patch("rdma_sweep.yaml", fake_yaml),
            ):
                with self.assertRaises(SystemExit) as ctx:
                    main()
                self.assertNotEqual(ctx.exception.code, 0)


# ---------------------------------------------------------------------------
# run_remote_result SSH paths  (4 scenarios)
# ---------------------------------------------------------------------------

class TestRunRemoteResultSSH(unittest.TestCase):
    """run_remote_result SSH execution branches.

    The SSH path is hard to test with real SSH, but the subprocess mocking
    covers the happy path, timeout, and exception branches.
    """

    def test_ssh_success(self):
        completed = subprocess.CompletedProcess(
            args=["ssh", "host", "cmd"],
            returncode=0, stdout="output\n", stderr="",
        )
        with (
            patch("rdma_remote._LOCAL_HOSTS", set()),  # not local → SSH
            patch("rdma_remote.subprocess.run", return_value=completed),
        ):
            result = run_remote_result("echo hi", "remote-host")

        self.assertTrue(result.ok)
        self.assertEqual(result.stdout, "output\n")

    def test_ssh_timeout(self):
        timeout = subprocess.TimeoutExpired(
            cmd=["ssh", "host", "cmd"],
            timeout=10,
            output=b"partial",
            stderr=b"",
        )
        with (
            patch("rdma_remote._LOCAL_HOSTS", set()),
            patch("rdma_remote.subprocess.run", side_effect=timeout),
        ):
            result = run_remote_result("sleep 100", "remote-host", timeout=10)

        self.assertFalse(result.ok)
        self.assertTrue(result.timed_out)
        self.assertEqual(result.returncode, 124)
        self.assertEqual(result.stdout, "partial")

    def test_ssh_exception(self):
        with (
            patch("rdma_remote._LOCAL_HOSTS", set()),
            patch("rdma_remote.subprocess.run", side_effect=RuntimeError("ssh: no route")),
        ):
            result = run_remote_result("echo hi", "remote-host")

        self.assertFalse(result.ok)
        self.assertEqual(result.exception, "ssh: no route")

    def test_ssh_includes_options_and_timeout(self):
        """SSH command includes connect timeout and configured options."""
        completed = subprocess.CompletedProcess(
            args=["ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes", "host", "cmd"],
            returncode=0,
        )
        with (
            patch("rdma_remote._LOCAL_HOSTS", set()),
            patch("rdma_remote.subprocess.run", return_value=completed) as run_mock,
        ):
            run_remote_result(
                "echo hi", "remote-host",
                ssh_config={"connect_timeout": 5, "sudo": False},
            )

        cmd = run_mock.call_args.args[0][0]
        self.assertEqual(cmd, "ssh")
        self.assertIn("-o", run_mock.call_args.args[0])
        self.assertIn("ConnectTimeout=5", run_mock.call_args.args[0])

    def test_ssh_options_none_does_not_crash(self):
        """run_remote_result with options=None (YAML null) does not TypeError."""
        completed = subprocess.CompletedProcess(
            args=["ssh", "-o", "ConnectTimeout=10", "host", "cmd"],
            returncode=0,
        )
        with (
            patch("rdma_remote._LOCAL_HOSTS", set()),
            patch("rdma_remote.init_local_hosts"),
            patch("rdma_remote.subprocess.run", return_value=completed) as run_mock,
        ):
            result = run_remote_result(
                "echo hi", "remote-host",
                ssh_config={"options": None, "sudo": False},
            )
        self.assertTrue(result.ok)
        # Verify no options leaked as None or empty
        cmd = run_mock.call_args.args[0]
        self.assertNotIn("BatchMode", cmd)
        self.assertNotIn("StrictHostKeyChecking", cmd)

    def test_ssh_dash_dash_before_host(self):
        """SSH args include '--' before the hostname (option injection guard)."""
        completed = subprocess.CompletedProcess(
            args=["ssh", "host", "cmd"],
            returncode=0,
        )
        with (
            patch("rdma_remote._LOCAL_HOSTS", set()),
            patch("rdma_remote.init_local_hosts"),
            patch("rdma_remote.subprocess.run", return_value=completed) as run_mock,
        ):
            run_remote_result("echo hi", "remote-host", ssh_config={"sudo": False})
        cmd = run_mock.call_args.args[0]
        self.assertIn("--", cmd, "SSH args must contain '--' before hostname")
        idx = cmd.index("--")
        self.assertGreater(idx, 0, "'--' must not be the first SSH arg")
        self.assertEqual(cmd[idx + 1], "remote-host",
                         "hostname must immediately follow '--'")


# ---------------------------------------------------------------------------
# _as_text helper  (3 scenarios)
# ---------------------------------------------------------------------------

class TestAsText(unittest.TestCase):
    """_as_text converts bytes/str/None to str for timeout output handling."""

    def test_none_returns_empty(self):
        from rdma_remote import _as_text
        self.assertEqual(_as_text(None), "")

    def test_str_passthrough(self):
        from rdma_remote import _as_text
        self.assertEqual(_as_text("hello"), "hello")

    def test_bytes_decoded_with_replace(self):
        from rdma_remote import _as_text
        self.assertEqual(_as_text(b"hello"), "hello")
        # Invalid UTF-8 → replacement chars, not crash
        self.assertEqual(_as_text(b"\xff\xfe"), "��")


# ---------------------------------------------------------------------------
# _cpu_core_keys : per-core-key filter (empty, aggregate excluded, mixed)
# ---------------------------------------------------------------------------

class TestCpuCoreKeys(unittest.TestCase):
    """_cpu_core_keys filter for per-core CPU data keys."""

    def test_empty_dict_returns_empty_list(self):
        from rdma_sweep import _cpu_core_keys
        self.assertEqual(_cpu_core_keys({}), [])

    def test_aggregate_cpu_excluded(self):
        """The 'cpu' aggregate line (no core number) is excluded."""
        from rdma_sweep import _cpu_core_keys
        self.assertEqual(
            sorted(_cpu_core_keys({"cpu": 1.0, "cpu0": 2.0, "cpu1": 3.0})),
            ["cpu0", "cpu1"],
        )

    def test_non_cpu_keys_excluded(self):
        """Keys not starting with 'cpu' are excluded."""
        from rdma_sweep import _cpu_core_keys
        self.assertEqual(
            sorted(_cpu_core_keys({"cpu0": 1.0, "other": 2.0, "cpu1": 3.0})),
            ["cpu0", "cpu1"],
        )

    def test_cpufreq_key_retained(self):
        """Keys starting with 'cpu' but not matching a core number are retained."""
        from rdma_sweep import _cpu_core_keys
        self.assertIn("cpufreq", _cpu_core_keys({"cpu0": 1.0, "cpufreq": 2.0}))


# ---------------------------------------------------------------------------
# _cpu_avg edge cases  (3 scenarios)
# ---------------------------------------------------------------------------

class TestCpuAvg(unittest.TestCase):
    """_cpu_avg and _core_sort_key helpers for per-core CPU metrics."""

    def test_empty_dict_returns_empty_string(self):
        from rdma_sweep import _cpu_avg
        self.assertEqual(_cpu_avg({}), "")

    def test_only_cpu_aggregate_skipped(self):
        """The 'cpu' aggregate line (no core number) is excluded."""
        from rdma_sweep import _cpu_avg
        result = _cpu_avg({"cpu": 50.0, "cpu0": 30.0, "cpu1": 70.0})
        self.assertEqual(result, 50.0)  # avg of 30 + 70 = 50

    def test_only_numbered_cores_counted(self):
        """Only cpu0/cpu1/etc are averaged; keys not starting with 'cpu' are excluded."""
        from rdma_sweep import _cpu_avg
        result = _cpu_avg({"cpu0": 30.0, "cpu1": 70.0, "other": 99.0})
        self.assertEqual(result, 50.0)  # 99.0 excluded (doesn't start with 'cpu')

    def test_core_sort_key_non_standard_returns_negative_one(self):
        """Non-standard cpu keys like 'cpufreq' fail int() and get sort key -1."""
        from rdma_sweep import _core_sort_key
        self.assertEqual(_core_sort_key("cpu0"), 0)
        self.assertEqual(_core_sort_key("cpu7"), 7)
        self.assertEqual(_core_sort_key("cpufreq"), -1)
        self.assertEqual(_core_sort_key("cpuinfo"), -1)


# ---------------------------------------------------------------------------
# generate_report edge cases  (2 scenarios)
# ---------------------------------------------------------------------------

class TestGenerateReportEdgeCases(unittest.TestCase):
    """generate_report handles missing result.json gracefully."""

    def test_missing_result_json_does_not_crash(self):
        """If _result_path points to a non-existent file, perf_data is empty."""
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            entry = {
                "BW_average": 100,
                "MsgRate": 1.0,
                "params": {"qp": 1},
            }
            (out / "summary.json").write_text(json.dumps([entry]))
            # No 0001/ directory — result.json missing

            generate_report(str(out))  # should not raise

            svg = (out / "chart.svg").read_text()
            self.assertIn("RDMA Perftest Sweep", svg)

    def test_run_config_report_overrides(self):
        """report section in run_config.json overrides chart title/subtitle."""
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            entry = {"BW_average": 100, "MsgRate": 1.0, "params": {"qp": 1}}
            (out / "summary.json").write_text(json.dumps([entry]))
            run_dir = out / "0001"
            run_dir.mkdir()
            (run_dir / "result.json").write_text(json.dumps({"_process": {"server_perf": {}}}))
            # Custom report config
            (out / "run_config.json").write_text(json.dumps({
                "report": {"title": "Custom Title", "subtitle": "Custom Subtitle"},
            }))

            generate_report(str(out))

            svg = (out / "chart.svg").read_text()
            self.assertIn("Custom Title", svg)
            self.assertIn("Custom Subtitle", svg)


    def test_svg_table_title_is_html_escaped(self):
        """_svg_table title must be HTML-escaped (all other SVG text is escaped)."""
        from rdma_sweep import _svg_table
        el = []
        _svg_table(el, ["QP", "cpu0"], [["1", "50.0"]], "Test <script>alert(1)</script>", 0, 0, 500)
        svg = "\n".join(el)
        self.assertNotIn("<script>", svg)
        self.assertIn("&lt;script&gt;", svg)

    def test_svg_table_empty_hdrs_renders_no_data_message(self):
        """_svg_table empty-hdrs guard (lines 1728-1730): renders 'no per-core data'
        message instead of crashing or rendering an empty table."""
        from rdma_sweep import _svg_table
        el = []
        _svg_table(el, [], [], "Per-Core CPU", 0, 0, 500)
        svg = "\n".join(el)
        self.assertIn("Per-Core CPU", svg)
        self.assertIn("no per-core data", svg)

    def test_cairosvg_subprocess_invoked_when_available(self):
        """generate_report invokes cairosvg subprocess when tool is found."""
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            entry = {"BW_average": 100, "MsgRate": 1.0, "params": {"qp": 1}}
            (out / "summary.json").write_text(json.dumps([entry]))
            with (
                patch("rdma_sweep.shutil.which", return_value="/usr/bin/cairosvg"),
                patch("rdma_sweep.subprocess.run") as run_mock,
            ):
                run_mock.return_value = subprocess.CompletedProcess(
                    args=[], returncode=0,
                )
                generate_report(str(out))

            # Only the cairosvg conversion call, no `which` subprocess
            calls = [c.args[0] for c in run_mock.call_args_list]
            self.assertEqual(len(calls), 1)
            self.assertIn("cairosvg", calls[0][0])

            svg = (out / "chart.svg").read_text()
            self.assertIn("RDMA Perftest Sweep", svg)

    def test_cairosvg_conversion_failure_handled_gracefully(self):
        """Exception handler: cairosvg conversion raises, caught
        by except Exception, 'PDF generation skipped' printed to stderr."""
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            entry = {"BW_average": 100, "MsgRate": 1.0, "params": {"qp": 1}}
            (out / "summary.json").write_text(json.dumps([entry]))

            def failing_cairosvg(*args, **kwargs):
                raise subprocess.CalledProcessError(1, args[0])

            with (
                patch("rdma_sweep.shutil.which", return_value="/usr/bin/cairosvg"),
                patch("rdma_sweep.subprocess.run", side_effect=failing_cairosvg),
                patch("sys.stderr", new_callable=io.StringIO) as mock_stderr,
            ):
                generate_report(str(out))

            self.assertIn("PDF generation skipped", mock_stderr.getvalue())
            self.assertIn("returned non-zero exit status", mock_stderr.getvalue())


class TestImportSideEffects(unittest.TestCase):
    """Module-level side effects during import."""

    def test_rdma_sweep_import_does_not_call_init_local_hosts(self):
        """Importing rdma_sweep must not trigger init_local_hosts()."""
        import subprocess as _sp, sys as _sys
        result = _sp.run(
            [_sys.executable, "-c", """
import sys, os
sys.path.insert(0, os.getcwd())
from rdma_remote import _LOCAL_HOSTS
assert len(_LOCAL_HOSTS) == 0, (
    f"rdma_remote import set _LOCAL_HOSTS: {_LOCAL_HOSTS}"
)
import rdma_sweep
from rdma_remote import _LOCAL_HOSTS as lh
assert len(lh) == 0, (
    f"rdma_sweep import triggered init_local_hosts(): {lh}"
)
print("OK: no import-time subprocess side effects")
"""],
            capture_output=True, text=True, timeout=15,
            cwd="/tmp/rdma-sweep-tool",
        )
        if result.returncode != 0:
            raise AssertionError(
                f"Import triggered init_local_hosts(): {result.stderr}"
            )


class TestWriteCsv(unittest.TestCase):
    """_write_csv renders the CSV with correct masquerade-trust gating."""

    def _rows(self, summary):
        import csv, tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "summary.csv"
            _write_csv(path, summary)
            with open(path, newline="") as f:
                return list(csv.reader(f))

    def _make_entry(self, **overrides):
        entry = {
            "server": {"host": "s1", "address": "10.0.0.2"},
            "client": {"host": "c1"},
            "params": {"qp": 1, "msg_size": "64K"},
            "BW_average": 42, "MsgRate": 5, "BW_peak": 44,
            "n_iterations": 100, "MsgSize": 65536, "t_avg": 0,
            # _write_csv recomputes cpu_avg from *_cpu_per_core (not these
            # top-level keys); they exist to be read by _svg_chart.
            "server_cpu_per_core": {"cpu0": 12.5},
            "client_cpu_per_core": {"cpu0": 8.3},
            "server_memory": {"MemUsed": 10, "MemUsedDelta": 5},
            "client_memory": {"MemUsed": 4, "MemUsedDelta": 2},
            "server_sys_ok": True, "client_sys_ok": True,
            "elapsed_sec": 10,
        }
        entry.update(overrides)
        return entry

    def _header_index(self, rows):
        return {name: i for i, name in enumerate(rows[0])}

    def test_empty_summary_creates_no_file(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "summary.csv"
            _write_csv(path, [])
            self.assertFalse(path.exists())

    def test_happy_path_renders_metric_values(self):
        rows = self._rows([self._make_entry()])
        self.assertEqual(len(rows), 2)  # header + 1 data row
        idx = self._header_index(rows)
        data = rows[1]
        self.assertEqual(data[idx["BW_average"]], "42")
        self.assertEqual(data[idx["MsgRate"]], "5")
        self.assertEqual(data[idx["error"]], "")
        # Sys-monitor columns render clean values when sys_ok is True
        self.assertEqual(data[idx["server_cpu_avg"]], "12.5")
        self.assertEqual(data[idx["server_mem_used"]], "10")
        self.assertEqual(data[idx["server_mem_used_delta"]], "5")

    def test_errored_entry_renders_ERR_for_all_perf_metrics(self):
        entry = self._make_entry(error="server crashed")
        rows = self._rows([entry])
        idx = self._header_index(rows)
        data = rows[1]
        self.assertEqual(data[idx["error"]], "server crashed")
        for key in ("BW_average", "MsgRate", "BW_peak", "n_iterations", "MsgSize", "t_avg"):
            self.assertEqual(data[idx[key]], "ERR",
                             f"{key} must be ERR for errored entry")

    def test_server_sys_ok_false_renders_ERR_on_server_columns(self):
        entry = self._make_entry(server_sys_ok=False)
        rows = self._rows([entry])
        idx = self._header_index(rows)
        data = rows[1]
        self.assertEqual(data[idx["server_cpu_avg"]], "ERR")
        self.assertEqual(data[idx["cpu_avg"]], "ERR")
        self.assertEqual(data[idx["server_mem_used"]], "ERR")
        self.assertEqual(data[idx["server_mem_used_delta"]], "ERR")
        # client side unaffected
        self.assertEqual(data[idx["client_cpu_avg"]], "8.3")
        self.assertEqual(data[idx["client_sys_ok"]], "True")

    def test_client_sys_ok_false_renders_ERR_on_client_columns(self):
        entry = self._make_entry(client_sys_ok=False)
        rows = self._rows([entry])
        idx = self._header_index(rows)
        data = rows[1]
        self.assertEqual(data[idx["client_cpu_avg"]], "ERR")
        self.assertEqual(data[idx["client_mem_used"]], "ERR")
        self.assertEqual(data[idx["client_mem_used_delta"]], "ERR")
        # server side unaffected
        self.assertEqual(data[idx["server_cpu_avg"]], "12.5")

    def test_both_sys_ok_false_renders_ERR_on_both_sides(self):
        entry = self._make_entry(server_sys_ok=False, client_sys_ok=False)
        rows = self._rows([entry])
        idx = self._header_index(rows)
        data = rows[1]
        self.assertEqual(data[idx["server_cpu_avg"]], "ERR")
        self.assertEqual(data[idx["client_cpu_avg"]], "ERR")

    def test_error_empty_string_falls_back_to_metric_values(self):
        # Empty string is falsy, so ``if entry.get("error"):`` treats it
        # as no-error.  Metrics render normally.
        entry = self._make_entry(error="")
        rows = self._rows([entry])
        idx = self._header_index(rows)
        data = rows[1]
        self.assertEqual(data[idx["BW_average"]], "42")

    def test_multiple_entries_with_different_params(self):
        e1 = self._make_entry(params={"qp": 1, "msg_size": "64K"})
        e2 = self._make_entry(params={"qp": 2, "msg_size": "1M", "inline": 1})
        rows = self._rows([e1, e2])
        idx = self._header_index(rows)
        self.assertIn("inline", idx)
        self.assertIn("msg_size", idx)
        self.assertIn("qp", idx)
        # e2 has inline=1, e1 has no inline key
        self.assertEqual(rows[2][idx["inline"]], "1")
        self.assertEqual(rows[1][idx["inline"]], "")

    def test_missing_server_client_dicts_do_not_crash(self):
        entry = self._make_entry()
        entry.pop("server", None)
        entry.pop("client", None)
        rows = self._rows([entry])
        idx = self._header_index(rows)
        data = rows[1]
        # str(None) would render "None" if .get returned None; "or {}" in the
        # production code forces "" instead.
        self.assertEqual(data[idx["server_host"]], "")
        self.assertEqual(data[idx["client_host"]], "")

    def test_missing_server_cpu_per_core_renders_empty(self):
        entry = self._make_entry()
        entry.pop("server_cpu_per_core", None)
        # _write_csv recomputes cpu_avg from the *_cpu_per_core dict; absent
        # → _cpu_avg({}) returns ""
        rows = self._rows([entry])
        idx = self._header_index(rows)
        self.assertEqual(rows[1][idx["server_cpu_avg"]], "")

    def test_cpu_per_core_backward_compat_fallback(self):
        # Old-format summary has ``cpu_per_core`` instead of the split
        # ``server_cpu_per_core``/``client_cpu_per_core``.  _write_csv falls
        # back to ``cpu_per_core`` for the server-side chart when the newer
        # key is absent.
        entry = self._make_entry()
        entry.pop("server_cpu_per_core", None)
        entry["cpu_per_core"] = {"cpu0": 15.0}
        rows = self._rows([entry])
        idx = self._header_index(rows)
        self.assertEqual(rows[1][idx["server_cpu_avg"]], "15.0")

    def test_absent_sys_ok_keys_default_to_true(self):
        # Pre-flag result files lack server_sys_ok/client_sys_ok; the CSV
        # must assume ok=True (render metric values, not ERR).
        entry = self._make_entry()
        entry.pop("server_sys_ok", None)
        entry.pop("client_sys_ok", None)
        rows = self._rows([entry])
        idx = self._header_index(rows)
        data = rows[1]
        self.assertEqual(data[idx["server_cpu_avg"]], "12.5")
        self.assertEqual(data[idx["client_cpu_avg"]], "8.3")

    def test_error_entry_with_stale_bw_renders_ERR_not_stale_value(self):
        # An errored entry that still carries BW_average at the top level
        # must render "ERR" in the CSV, never the stale number.
        entry = self._make_entry(error="connection refused", BW_average=999)
        rows = self._rows([entry])
        idx = self._header_index(rows)
        self.assertEqual(rows[1][idx["BW_average"]], "ERR")


class TestValidatePerftestMetrics(unittest.TestCase):
    """_validate_perftest metrics correctly distinguishes valid/invalid results."""

    def test_bw_both_metrics_present_returns_ok(self):
        result = {"results": {"BW_average": 27.38, "MsgRate": 5.0}}
        self.assertEqual(_validate_perftest_metrics("ib_write_bw", result), "")

    def test_bw_missing_MsgRate_returns_error(self):
        result = {"results": {"BW_average": 27.38}}
        error = _validate_perftest_metrics("ib_write_bw", result)
        self.assertIn("MsgRate", error)

    def test_bw_missing_BW_average_returns_error(self):
        result = {"results": {"MsgRate": 5.0}}
        error = _validate_perftest_metrics("ib_write_bw", result)
        self.assertIn("BW_average", error)

    def test_bw_missing_both_returns_error_with_both_names(self):
        result = {"results": {}}
        error = _validate_perftest_metrics("ib_write_bw", result)
        self.assertIn("BW_average", error)
        self.assertIn("MsgRate", error)

    def test_latency_t_avg_present_returns_ok(self):
        result = {"results": {"t_avg": 1.2}}
        self.assertEqual(_validate_perftest_metrics("ib_write_lat", result), "")

    def test_latency_missing_t_avg_returns_error(self):
        result = {"results": {}}
        error = _validate_perftest_metrics("ib_write_lat", result)
        self.assertIn("t_avg", error)

    def test_latency_with_extra_metrics_accepts_t_avg_only(self):
        # *_lat binary only requires t_avg; BW_average/MsgRate are irrelevant
        result = {"results": {"t_avg": 1.2, "BW_average": 0}}
        self.assertEqual(_validate_perftest_metrics("ib_write_lat", result), "")

    def test_result_has_error_short_circuits(self):
        result = {"error": "simulated"}
        self.assertEqual(_validate_perftest_metrics("ib_write_bw", result), "")

    def test_results_key_missing_returns_error(self):
        result = {"result": [{"BW_average": 27.38}]}
        self.assertIn("missing results object",
                      _validate_perftest_metrics("ib_write_bw", result))

    def test_results_is_not_dict_returns_error(self):
        result = {"results": [{"BW_average": 27.38}]}
        self.assertIn("missing results object",
                      _validate_perftest_metrics("ib_write_bw", result))

    def test_results_is_none_returns_error(self):
        result = {"results": None}
        self.assertIn("missing results object",
                      _validate_perftest_metrics("ib_write_bw", result))

    def test_bw_zero_value_accepted(self):
        """Zero bandwidth/rate passes validation (perftest may report 0 for a
        genuine measurement).  This is a documented spec choice: the CSV
        renders the number faithfully, and perftest's non-zero exit code (not
        validation) gates the error flag.  Change here must be deliberate."""
        result = {"results": {"BW_average": 0.0, "MsgRate": 0.0}}
        self.assertEqual(_validate_perftest_metrics("ib_write_bw", result), "")

    def test_latency_t_avg_explicitly_none_returns_error(self):
        result = {"results": {"t_avg": None}}
        self.assertIn("t_avg", _validate_perftest_metrics("ib_write_lat", result))


class TestMainEntryPoint(unittest.TestCase):
    """``__name__ == '__main__'`` guard (line 1835): runs main() on direct execution."""

    def test_cli_help_exits_ok(self):
        """Running ``rdma_sweep.py --help`` as a script reaches ``__name__`` guard."""
        result = subprocess.run(
            [sys.executable, "rdma_sweep.py", "--help"],
            capture_output=True, text=True, timeout=10,
            cwd="/tmp/rdma-sweep-tool",
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("usage", result.stdout.lower())

    # ------------------------------------------------------------------ #
    #  Error exit code (line ~1831)
    # ------------------------------------------------------------------ #

    @patch("rdma_sweep.init_local_hosts")
    @patch("rdma_sweep.run_sweep")
    @patch("rdma_sweep.sys.exit")
    def test_main_exits_nonzero_on_sweep_errors(
        self, mock_exit, mock_run, mock_init,
    ):
        """main() exits 1 when summary.json contains error entries."""
        mock_run.return_value = []

        tmpdir = tempfile.mkdtemp()
        try:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text("sweep:\n  - name: qp\n    values: [1]\n")

            summary_path = Path(tmpdir) / "summary.json"
            summary_path.write_text(json.dumps([
                {"error": "connection timeout", "params": {"qp": 1}},
                {"error": None, "params": {"qp": 2}},
            ]))

            with patch("sys.argv", ["rdma_sweep.py", "--config", str(config_path),
                                    "--output-dir", tmpdir]):
                main()

            mock_exit.assert_called_once_with(1)
        finally:
            shutil.rmtree(tmpdir)

    @patch("rdma_sweep.init_local_hosts")
    @patch("rdma_sweep.run_sweep")
    @patch("rdma_sweep.sys.exit")
    def test_main_exits_zero_when_no_sweep_errors(
        self, mock_exit, mock_run, mock_init,
    ):
        """main() exits 0 when summary.json has no error entries."""
        mock_run.return_value = []

        tmpdir = tempfile.mkdtemp()
        try:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text("sweep:\n  - name: qp\n    values: [1]\n")

            summary_path = Path(tmpdir) / "summary.json"
            summary_path.write_text(json.dumps([
                {"error": None, "params": {"qp": 1}},
            ]))

            with patch("sys.argv", ["rdma_sweep.py", "--config", str(config_path),
                                    "--output-dir", tmpdir]):
                main()

            mock_exit.assert_not_called()
        finally:
            shutil.rmtree(tmpdir)

    @patch("rdma_sweep.init_local_hosts")
    @patch("rdma_sweep.run_sweep")
    @patch("rdma_sweep.sys.exit")
    def test_main_exits_zero_when_no_summary_file(
        self, mock_exit, mock_run, mock_init,
    ):
        """main() exits normally when summary.json does not exist."""
        mock_run.return_value = []

        tmpdir = tempfile.mkdtemp()
        try:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text("sweep:\n  - name: qp\n    values: [1]\n")

            with patch("sys.argv", ["rdma_sweep.py", "--config", str(config_path),
                                    "--output-dir", tmpdir]):
                main()

            mock_exit.assert_not_called()
        finally:
            shutil.rmtree(tmpdir)

    @patch("rdma_sweep.init_local_hosts")
    @patch("rdma_sweep.run_sweep")
    @patch("rdma_sweep.sys.exit")
    def test_main_exits_zero_on_corrupt_summary(
        self, mock_exit, mock_run, mock_init,
    ):
        """main() exits normally when summary.json is corrupt JSON."""
        mock_run.return_value = []

        tmpdir = tempfile.mkdtemp()
        try:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text("sweep:\n  - name: qp\n    values: [1]\n")

            summary_path = Path(tmpdir) / "summary.json"
            summary_path.write_text("not valid json")

            with patch("sys.argv", ["rdma_sweep.py", "--config", str(config_path),
                                    "--output-dir", tmpdir]):
                main()

            mock_exit.assert_not_called()
        finally:
            shutil.rmtree(tmpdir)

    # ------------------------------------------------------------------ #
    #  Dry-run (--dry-run / -n)
    # ------------------------------------------------------------------ #

    def test_dry_run_cli_flag(self):
        """``--dry-run`` lists combos and exits 0 without SSH."""
        tmpdir = tempfile.mkdtemp()
        try:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text("sweep:\n  - name: qp\n    values: [1, 2]\n")

            result = subprocess.run(
                [sys.executable, "rdma_sweep.py", "--config", str(config_path),
                 "--dry-run"],
                capture_output=True, text=True, timeout=10,
                cwd="/tmp/rdma-sweep-tool",
            )
            self.assertEqual(result.returncode, 0)
            self.assertIn("qp=1", result.stdout)
            self.assertIn("qp=2", result.stdout)
            self.assertIn("Total:", result.stdout)
        finally:
            shutil.rmtree(tmpdir)

    def test_dry_run_print_empty_config(self):
        """_dry_run_print handles empty sweep gracefully (1 combo with no params)."""
        config = {"sweep": []}
        with patch("sys.stdout", new_callable=io.StringIO) as mock_out:
            _dry_run_print(config)
        # sweep_config yields fixed={} even when sweep is empty
        self.assertIn("Total: 1 combo(s)", mock_out.getvalue())


# ---------------------------------------------------------------------------
# _parse_proc_stat / _parse_proc_softirqs / _parse_proc_meminfo
# ---------------------------------------------------------------------------

class TestParseProcStat(unittest.TestCase):
    """_parse_proc_stat extracts per-core CPU tick dicts from /proc/stat lines."""

    def test_parses_cpu_lines(self):
        lines = [
            "cpu  1 2 3 4 5 6 7 8",
            "cpu0 1 2 3 4 5 6 7 8",
            "cpu1 9 8 7 6 5 4 3 2",
        ]
        result = _parse_proc_stat(lines)
        self.assertIn("cpu", result)
        self.assertIn("cpu0", result)
        self.assertIn("cpu1", result)
        self.assertEqual(result["cpu0"]["user"], 1)
        self.assertEqual(result["cpu0"]["idle"], 4)
        self.assertEqual(result["cpu0"]["steal"], 8)
        self.assertEqual(result["cpu1"]["system"], 7)

    def test_skips_non_cpu_lines(self):
        lines = [
            "cpu0 1 2 3 4 5 6 7 8",
            "intr 12345",
            "ctxt 999",
            "cpu1 1 2 3 4 5 6 7 8",
        ]
        result = _parse_proc_stat(lines)
        self.assertIn("cpu0", result)
        self.assertIn("cpu1", result)
        self.assertEqual(len(result), 2)

    def test_skips_malformed_numeric_line(self):
        lines = [
            "cpu0 1 2 BAD 4 5 6 7 8",
            "cpu1 1 2 3 4 5 6 7 8",
        ]
        result = _parse_proc_stat(lines)
        self.assertNotIn("cpu0", result)
        self.assertIn("cpu1", result)

    def test_handles_blank_lines(self):
        lines = [
            "cpu0 1 2 3 4 5 6 7 8",
            "",
            "cpu1 1 2 3 4 5 6 7 8",
        ]
        result = _parse_proc_stat(lines)
        self.assertIn("cpu0", result)
        self.assertIn("cpu1", result)
        self.assertEqual(len(result), 2)

    def test_empty_input_returns_empty_dict(self):
        self.assertEqual(_parse_proc_stat([]), {})

    def test_skips_cpu_line_with_too_few_fields(self):
        """Line with < 8 values would cause silent zip truncation — skip it."""
        lines = [
            "cpu0 1 2 3 4 5",          # only 5 values
            "cpu1 1 2 3 4 5 6 7 8",    # normal
        ]
        result = _parse_proc_stat(lines)
        self.assertNotIn("cpu0", result)
        self.assertIn("cpu1", result)
        self.assertEqual(len(result), 1)


class TestParseProcSoftirqs(unittest.TestCase):
    """_parse_proc_softirqs extracts per-type summed IRQ counts."""

    def test_parses_single_line(self):
        lines = ["NET_RX:  12345  0  67890  0"]
        result = _parse_proc_softirqs(lines)
        self.assertEqual(result["NET_RX"], 12345 + 67890)

    def test_parses_skips_cpu_header(self):
        lines = [
            "          CPU0 CPU1 CPU2",
            "NET_RX:  123   0    456",
        ]
        result = _parse_proc_softirqs(lines)
        self.assertNotIn("CPU", result)
        self.assertEqual(result["NET_RX"], 123 + 0 + 456)

    def test_skips_malformed_numeric_line(self):
        lines = [
            "NET_RX:  1 2 3",
            "BAD_IRQ: oops 12",
        ]
        result = _parse_proc_softirqs(lines)
        self.assertIn("NET_RX", result)
        self.assertNotIn("BAD_IRQ", result)

    def test_handles_blank_lines(self):
        lines = [
            "NET_RX:  1 2 3",
            "",
            "TIMER:  4 5 6",
        ]
        result = _parse_proc_softirqs(lines)
        self.assertIn("NET_RX", result)
        self.assertIn("TIMER", result)

    def test_empty_input_returns_empty_dict(self):
        self.assertEqual(_parse_proc_softirqs([]), {})


class TestParseProcMeminfo(unittest.TestCase):
    """_parse_proc_meminfo extracts key→kB dict from /proc/meminfo lines."""

    def test_parses_basic_lines(self):
        lines = [
            "MemTotal: 1048576 kB",
            "MemFree: 524288 kB",
            "Buffers: 16384 kB",
        ]
        result = _parse_proc_meminfo(lines)
        self.assertEqual(result["MemTotal"], 1048576)
        self.assertEqual(result["MemFree"], 524288)
        self.assertEqual(result["Buffers"], 16384)

    def test_skips_line_with_no_colon(self):
        lines = [
            "MemTotal: 100 kB",
            "some garbage text without colon",
            "MemFree: 40 kB",
        ]
        result = _parse_proc_meminfo(lines)
        self.assertIn("MemTotal", result)
        self.assertIn("MemFree", result)
        self.assertEqual(len(result), 2)

    def test_skips_key_with_no_value(self):
        lines = [
            "MemTotal: 100 kB",
            "SomeKey:",
            "MemFree: 40 kB",
        ]
        result = _parse_proc_meminfo(lines)
        self.assertIn("MemTotal", result)
        self.assertNotIn("SomeKey", result)

    def test_skips_non_numeric_value(self):
        lines = [
            "MemTotal: 100 kB",
            "BadVal: notanumber kB",
            "MemFree: 40 kB",
        ]
        result = _parse_proc_meminfo(lines)
        self.assertIn("MemTotal", result)
        self.assertNotIn("BadVal", result)

    def test_strips_key_whitespace(self):
        lines = ["  MemTotal: 100 kB"]
        result = _parse_proc_meminfo(lines)
        self.assertIn("MemTotal", result)

    def test_empty_input_returns_empty_dict(self):
        self.assertEqual(_parse_proc_meminfo([]), {})


# ---------------------------------------------------------------------------
# _scale_y  (3 scenarios)
# ---------------------------------------------------------------------------

class TestScaleY(unittest.TestCase):
    """_scale_y converts data values to pixel y-coordinates."""

    def test_maps_min_to_bottom(self):
        # cy=200, ch=100 → bottom y = 200 + 100 = 300
        result = _scale_y(0, 200, 100, 0, 100)
        self.assertEqual(result, 300.0)

    def test_maps_max_to_top(self):
        result = _scale_y(100, 200, 100, 0, 100)
        self.assertEqual(result, 200.0)

    def test_maps_midpoint(self):
        result = _scale_y(50, 200, 100, 0, 100)
        self.assertEqual(result, 250.0)


# ---------------------------------------------------------------------------
# _run_one_combo  (2 scenarios)
# ---------------------------------------------------------------------------

class TestRunOneCombo(unittest.TestCase):
    """_run_one_combo runs one sweep combo and writes the result."""

    def _runtime(self) -> dict[str, Any]:
        return {
            "test": "ib_write_bw",
            "duration": 2,
            "use_gpu": False,
            "server": {"host": "srv", "address": "10.0.0.2"},
            "client": {"host": "cli"},
            "perftest": {"dir": "/opt/perftest", "perf_record": True, "wait_timeout": 30,
                         "default_port": 18515, "env": {}},
            "ssh": {"sudo": True, "allow_local": False, "connect_timeout": 10, "options": []},
            "report": {"title": "RDMA Sweep", "subtitle": ""},
        }

    def _clean_grab(self) -> dict[str, Any]:
        return {
            "time": "2026-06-28",
            "cores": {"cpu0": {"user": 0, "nice": 0, "system": 0, "idle": 1000, "iowait": 0}},
            "mem_kB": {"MemTotal": 1048576, "MemFree": 524288, "Buffers": 0, "Cached": 0},
            "softirqs": {"NET_RX": 100},
        }

    def test_happy_path_runs_perftest_and_saves_result(self):
        combo = {"msg_size": 64, "qp": 1}
        monitor = MagicMock(spec=SysMonitor)
        monitor.grab.return_value = self._clean_grab()
        result_files: list[Path] = []

        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp)
            with (
                patch("rdma_sweep._build_args", return_value=["-q", "1"]),
                patch("rdma_sweep.run_perftest", return_value={
                    "results": {"BW_average": 100.0, "MsgRate": 1.0},
                    "_process": {"server_perf": {}, "client_usage": {}},
                }),
            ):
                result_path = _run_one_combo(
                    1, combo, self._runtime(), monitor, monitor, out_path, result_files,
                )

            self.assertTrue(result_path.exists())
            self.assertIn(result_path, result_files)
            data = json.loads(result_path.read_text())
            self.assertEqual(data["results"]["BW_average"], 100.0)
            self.assertIn("_meta", data)
            self.assertEqual(data["_meta"]["duration"], 2)
            # a clean run (no top-level error, no cleanup/perf-cleanup error) must
            # NOT get a fabricated run_error — the promotion else-branch is gated.
            self.assertNotIn("run_error", data["_meta"])

    def test_error_perftest_propagates_to_result(self):
        combo = {"msg_size": 64, "qp": 4}
        monitor = MagicMock(spec=SysMonitor)
        monitor.grab.return_value = self._clean_grab()
        result_files: list[Path] = []

        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp)
            with (
                patch("rdma_sweep._build_args", return_value=["-q", "4"]),
                patch("rdma_sweep.run_perftest", return_value={
                    "error": "client run failed: exit 2",
                    "_process": {"server_perf": {}},
                }),
            ):
                result_path = _run_one_combo(
                    1, combo, self._runtime(), monitor, monitor, out_path, result_files,
                )

            data = json.loads(result_path.read_text())
            self.assertEqual(data["error"], "client run failed: exit 2")
            self.assertIn("run_error", data["_meta"])

    def test_perf_cleanup_error_promoted_to_run_error_keeps_bw(self):
        # A run with valid BW but a LEAKED perf sampler (perf_cleanup_error) has
        # no top-level "error", so without promotion it would pass silently with
        # exit 0 while a privileged sampler contaminates the NEXT combo's CPU
        # attribution.  _save_combo_result must promote it to _meta.run_error so
        # it surfaces in the summary error column, is counted by
        # _count_summary_errors, and forces a non-zero exit — WITHOUT discarding
        # this run's already-valid bandwidth measurement.
        combo = {"msg_size": 64, "qp": 4}
        monitor = MagicMock(spec=SysMonitor)
        monitor.grab.return_value = self._clean_grab()
        result_files: list[Path] = []

        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp)
            with (
                patch("rdma_sweep._build_args", return_value=["-q", "4"]),
                patch("rdma_sweep.run_perftest", return_value={
                    "results": {"BW_average": 100.0, "MsgRate": 1.0},
                    "_process": {
                        "server_perf": {},
                        "perf_cleanup_error": "perf survived SIGKILL on pid 4242",
                    },
                }),
            ):
                result_path = _run_one_combo(
                    1, combo, self._runtime(), monitor, monitor, out_path, result_files,
                )

            data = json.loads(result_path.read_text())
            # valid BW is preserved (no top-level error swallows it) ...
            self.assertNotIn("error", data)
            self.assertEqual(data["results"]["BW_average"], 100.0)
            # ... but the leak is promoted to a loud, attributed run_error ...
            self.assertTrue(data["_meta"]["run_error"].startswith("cleanup_error:"))
            self.assertIn("perf survived SIGKILL on pid 4242", data["_meta"]["run_error"])
            # ... which the summary surfaces as a counted error while still
            # carrying the valid BW value (flagged, not erased).
            entry = _summary_entry(data["_meta"], data["results"])
            self.assertTrue(entry["error"])
            self.assertEqual(entry["BW_average"], 100.0)

    def test_failed_server_grab_sets_sys_after_ok_false(self):
        """A failed SysMonitor.grab() flows through _build_run_metadata and sets
        server_sys_after_ok=False in the written result JSON.  This pins the
        source-of-truth line so a regression (e.g. flipping the 'not in' check)
        would break this test while leaving all other grab-mocked tests green."""
        combo = {"msg_size": 64, "qp": 2}
        monitor = MagicMock(spec=SysMonitor)
        clean = self._clean_grab()
        error_grab = {"error": "ssh timeout"}
        # grab call order: before-server, before-client, after-server (fails), after-client
        monitor.grab.side_effect = [clean, clean, error_grab, clean]
        result_files: list[Path] = []

        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp)
            with (
                patch("rdma_sweep._build_args", return_value=["-q", "2"]),
                patch("rdma_sweep.run_perftest", return_value={
                    "results": {"BW_average": 50.0, "MsgRate": 0.5},
                    "_process": {"server_perf": {}, "client_usage": {}},
                }),
            ):
                result_path = _run_one_combo(
                    1, combo, self._runtime(), monitor, monitor, out_path, result_files,
                )

            data = json.loads(result_path.read_text())
            self.assertIs(data["_meta"]["server_sys_after_ok"], False)
            self.assertIs(data["_meta"]["client_sys_after_ok"], True)


class TestLoadPerfBarSeriesErrorPaths(unittest.TestCase):
    """_load_perf_bar_series handles corrupt/missing result.json without crashing."""

    def test_corrupt_json_returns_none_sentinel(self):
        """Invalid JSON in result.json produces a None sentinel (profile unavailable),
        distinct from {} (perf disabled/no consumers)."""
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            run_dir = out / "0001"
            run_dir.mkdir()
            (run_dir / "result.json").write_text("not valid json {{{")
            entry = {
                "BW_average": 100,
                "MsgRate": 1.0,
                "params": {"qp": 1},
                "_result_path": str(run_dir / "result.json"),
            }
            perf_data, _syms, _series = _load_perf_bar_series([entry], [0])
            self.assertEqual(len(perf_data), 1)
            self.assertIsNone(perf_data[0])


class TestLaunchServerAndGetPidErrorPaths(unittest.TestCase):
    """_launch_server_and_get_pid returns error when remote PID read fails."""

    def test_server_pid_read_failure_returns_error(self):
        """If the remote PID file cannot be read, return (None, error_msg)."""
        process: dict[str, Any] = {"commands": {}}
        calls: list[int] = []

        def fake_run(cmd, host, ssh_config=None, sudo=None):
            calls.append(1)
            # First call = server launch → succeed
            if len(calls) == 1:
                return RemoteResult(host=host, command=cmd, stdout="")
            # Second call = 'cat {pid}' → fail
            return RemoteResult(host=host, command=cmd, returncode=1)

        with (
            patch("rdma_sweep._run_remote_result", side_effect=fake_run),
            patch("rdma_sweep._wait_for_port"),
        ):
            pid, err = _launch_server_and_get_pid(
                tmp_dir_q="/tmp/test",
                server_pid_q="/tmp/test.pid",
                server_log_q="/tmp/test.log",
                server_cmd="/opt/perftest/ib_write_bw --server",
                run_id="test-run",
                server_host="server-host",
                ssh_config={"sudo": True},
                extra_args=["-p", "12345"],
                default_port=18515,
                wait_timeout=10,
                process=process,
            )

        self.assertIsNone(pid)
        self.assertIsNotNone(err)
        self.assertIn("failed to read server PID", err)
        self.assertIsNone(process["server_pid"])
        self.assertIn("server_start", process["commands"])
        self.assertIn("server_pid_read", process["commands"])


class TestReadClientTimeUsageErrorPaths(unittest.TestCase):
    """_read_client_time_usage returns empty dict on SSH failure."""

    def test_ssh_failure_returns_empty_dict(self):
        """When _run_remote_result fails, return {} without processing stdout."""
        process: dict[str, Any] = {"commands": {}}
        with patch("rdma_sweep._run_remote_result") as mock_run:
            mock_run.return_value = RemoteResult(
                host="test-host", command="cat /tmp/time.out",
                returncode=1, stderr="ssh: Connection refused",
            )
            result = _read_client_time_usage(
                time_q="/tmp/time.out",
                client_host="test-host",
                ssh_config=None,
                process=process,
            )

        self.assertEqual(result, {})
        self.assertIn("commands", process)
        self.assertIn("client_time_read", process["commands"])


class TestEnrichErrorWithServerLog(unittest.TestCase):
    """_enrich_error_with_server_log fetches server log tail on error, no-op otherwise."""

    def test_noop_when_no_error(self):
        """No error key → no server log fetch, no side effects."""
        process: dict[str, Any] = {"commands": {}}
        result: dict[str, Any] = {"_process": process}
        with patch("rdma_sweep._fetch_server_log_tail") as mock_fetch:
            _enrich_error_with_server_log(result, "host", "/tmp/srv.log", None)
        mock_fetch.assert_not_called()
        self.assertNotIn("server_log_tail", process)

    def test_noop_when_no_process(self):
        """Error key but no _process → no crash, no fetch."""
        result: dict[str, Any] = {"error": "oops"}
        with patch("rdma_sweep._fetch_server_log_tail") as mock_fetch:
            _enrich_error_with_server_log(result, "host", "/tmp/srv.log", None)
        mock_fetch.assert_not_called()

    def test_fetches_log_tail_on_error(self):
        """Error key present → fetches server log and attaches to process."""
        process: dict[str, Any] = {"commands": {}}
        result: dict[str, Any] = {"error": "client failed", "_process": process}
        with patch("rdma_sweep._fetch_server_log_tail") as mock_fetch:
            mock_fetch.return_value = (
                "[2026-06-28 14:00:00] rdma_get_cm_event: No such device"
            )
            _enrich_error_with_server_log(result, "host", "/tmp/srv.log", None)

        mock_fetch.assert_called_once()
        self.assertIn("server_log_tail", process)
        self.assertIn("No such device", process["server_log_tail"])

    def test_empty_log_tail_not_attached(self):
        """Empty log tail string is not attached to process."""
        process: dict[str, Any] = {"commands": {}}
        result: dict[str, Any] = {"error": "client failed", "_process": process}
        with patch("rdma_sweep._fetch_server_log_tail") as mock_fetch:
            mock_fetch.return_value = ""
            _enrich_error_with_server_log(result, "host", "/tmp/srv.log", None)

        mock_fetch.assert_called_once()
        self.assertNotIn("server_log_tail", process)

    def test_catches_fetch_exception(self):
        """Exception from _fetch_server_log_tail is recorded as error, not propagated."""
        process: dict[str, Any] = {"commands": {}}
        result: dict[str, Any] = {"error": "client failed", "_process": process}
        with patch("rdma_sweep._fetch_server_log_tail") as mock_fetch:
            mock_fetch.side_effect = RuntimeError("SSH connection lost")
            _enrich_error_with_server_log(result, "host", "/tmp/srv.log", None)

        mock_fetch.assert_called_once()
        self.assertIn("server_log_tail_error", process)
        self.assertEqual(process["server_log_tail_error"], "SSH connection lost")
        self.assertNotIn("server_log_tail", process)

    def test_commands_recorded_in_process(self):
        """Underlying _fetch_server_log_tail records commands in process."""
        process: dict[str, Any] = {"commands": {}}
        result: dict[str, Any] = {"error": "client failed", "_process": process}
        with patch("rdma_sweep._fetch_server_log_tail") as mock_fetch:
            mock_fetch.return_value = "server error log"
            _enrich_error_with_server_log(result, "host", "/tmp/srv.log", {"sudo": True})

        mock_fetch.assert_called_once_with(
            "host", "/tmp/srv.log", {"sudo": True}, process,
        )


class TestChartColumnPos(unittest.TestCase):
    """_chart_column_pos computes column positions for chart multi-column layouts."""

    def test_single_column(self):
        """One column fills the content area."""
        cx, cw = _chart_column_pos(0, 1, 10, 100)
        self.assertEqual(cx, 10)
        self.assertEqual(cw, 80)  # (100 - 2*10) / 1 = 80

    def test_two_columns_first(self):
        """First of two columns starts at the left margin."""
        cx, cw = _chart_column_pos(0, 2, 10, 100)
        self.assertEqual(cx, 10)
        self.assertEqual(cw, 35)  # (100 - 3*10) / 2 = 35

    def test_two_columns_second(self):
        """Second of two columns starts after the first column + gap."""
        cx, cw = _chart_column_pos(1, 2, 10, 100)
        self.assertEqual(cx, 55)  # 10 + 1 * (35 + 10)
        self.assertEqual(cw, 35)

    def test_three_columns(self):
        """Three columns with 10px margins."""
        cx0, cw = _chart_column_pos(0, 3, 10, 100)
        self.assertEqual(cw, 20)  # (100 - 4*10) / 3 = 20
        self.assertEqual(cx0, 10)
        cx1, _ = _chart_column_pos(1, 3, 10, 100)
        self.assertEqual(cx1, 40)  # 10 + 1 * (20 + 10)
        cx2, _ = _chart_column_pos(2, 3, 10, 100)
        self.assertEqual(cx2, 70)  # 10 + 2 * (20 + 10)

    def test_no_margin(self):
        """Zero margin means columns touch each other."""
        cx, cw = _chart_column_pos(0, 2, 0, 100)
        self.assertEqual(cx, 0)
        self.assertEqual(cw, 50)  # (100 - 0) / 2 = 50

    def test_last_column_fits_within_total_width(self):
        """Right edge of the last column does not exceed total_width."""
        num_cols = 4
        margin = 8
        total = 200
        _, cw = _chart_column_pos(0, num_cols, margin, total)
        cx_last, cw_last = _chart_column_pos(num_cols - 1, num_cols, margin, total)
        right_edge = cx_last + cw_last
        self.assertAlmostEqual(right_edge, total - margin)
        self.assertEqual(cw, cw_last)  # all columns equal width


class TestMakeProcessTracker(unittest.TestCase):
    """_make_process_tracker creates the per-run process tracking dict."""

    def test_returns_expected_keys(self):
        """All expected keys are present with correct initial values."""
        tracker = _make_process_tracker(
            run_id="abc123def456",
            server_host="srv-host",
            client_host="cli-host",
            server_address="10.0.0.2",
        )
        self.assertEqual(tracker["run_id"], "abc123def456")
        self.assertEqual(tracker["server_host"], "srv-host")
        self.assertEqual(tracker["client_host"], "cli-host")
        self.assertEqual(tracker["server_address"], "10.0.0.2")
        self.assertIsNone(tracker["server_pid"])
        self.assertEqual(tracker["server_perf"], {})
        self.assertEqual(tracker["client_usage"], {})
        self.assertEqual(tracker["commands"], {})

    def test_returns_new_dict_each_call(self):
        """Each call returns an independent dict (no shared mutable state)."""
        a = _make_process_tracker("r1", "h1", "h2", "addr")
        b = _make_process_tracker("r2", "h3", "h4", "addr2")
        a["commands"]["test"] = "mutated"
        self.assertNotIn("test", b["commands"])


class TestMakeServerErrorResult(unittest.TestCase):
    """_make_server_error_result creates error dict with server log enrichment."""

    def test_returns_error_dict_with_server_log_tail(self):
        """Returns {"error": ..., "_process": ...} with server_log_tail added."""
        process: dict[str, Any] = {"commands": {}}
        with patch("rdma_sweep._fetch_server_log_tail", return_value="log tail content"):
            result = _make_server_error_result(
                "server crashed", "host1", "/tmp/srv.log", None, process,
            )

        self.assertEqual(result["error"], "server crashed")
        self.assertIs(result["_process"], process)
        self.assertEqual(process.get("server_log_tail"), "log tail content")

    def test_enrichment_failure_does_not_mask_error(self):
        """A failed server log fetch sets server_log_tail_error, error is intact."""
        process: dict[str, Any] = {"commands": {}}
        with patch("rdma_sweep._fetch_server_log_tail", side_effect=IOError("no log")):
            result = _make_server_error_result(
                "timeout", "host2", "/tmp/srv.log", None, process,
            )

        self.assertEqual(result["error"], "timeout")
        self.assertIn("server_log_tail_error", process)

    def test_always_constructs_error_key(self):
        """Error key is always set in the returned dict."""
        process: dict[str, Any] = {"commands": {}}
        with patch("rdma_sweep._fetch_server_log_tail", return_value="ok"):
            result = _make_server_error_result(
                "fail", "host3", "/tmp/srv.log", None, process,
            )
        self.assertEqual(result["error"], "fail")

    def test_handle_run_exception_calls_it(self):
        """_handle_run_exception delegates to _make_server_error_result."""
        process: dict[str, Any] = {"commands": {}}
        with patch("rdma_sweep._fetch_server_log_tail", return_value="log"):
            result = _handle_run_exception(
                ValueError("bad"), "host4", "/tmp/srv.log", None, process,
            )
        self.assertEqual(result["error"], "bad")
        self.assertEqual(process["exception"], "bad")


class TestConfigInt(unittest.TestCase):
    """_config_int extracts integer config values with type conversion."""

    def test_returns_int_when_key_present(self):
        """Existing int key returns its value."""
        result = _config_int({"timeout": 30}, "timeout", 10)
        self.assertEqual(result, 30)

    def test_converts_string_to_int(self):
        """String value is converted to int."""
        result = _config_int({"timeout": "45"}, "timeout", 10)
        self.assertEqual(result, 45)

    def test_returns_default_when_key_absent(self):
        """Missing key returns the default."""
        result = _config_int({}, "timeout", 60)
        self.assertEqual(result, 60)

    def test_returns_default_when_value_is_none(self):
        """None value returns the default (same as _config_val behavior)."""
        result = _config_int({"timeout": None}, "timeout", 30)
        self.assertEqual(result, 30)

    def test_default_zero_when_omitted(self):
        """Omitting default uses 0."""
        result = _config_int({"threads": None}, "threads")
        self.assertEqual(result, 0)


class TestChartContentBox(unittest.TestCase):
    """_chart_content_box computes inset chart content area."""

    def test_basic_inset(self):
        self.assertEqual(_chart_content_box(0, 0, 100, 100), (8, 4, 84, 92))

    def test_with_offset(self):
        self.assertEqual(_chart_content_box(50, 30, 200, 50), (58, 34, 184, 42))

    def test_typical_panel_values(self):
        """Matches the 220-high panel pattern used by _line_chart / _multi_line_chart."""
        self.assertEqual(_chart_content_box(100, 200, 500, 220), (108, 204, 484, 212))

    def test_stacked_bar_panel(self):
        """Matches the 250-high panel pattern used by _stacked_bar."""
        self.assertEqual(_chart_content_box(16, 500, 1148, 250), (24, 504, 1132, 242))

    def test_negative_coordinates(self):
        self.assertEqual(_chart_content_box(-10, -5, 100, 80), (-2, -1, 84, 72))

    def test_zero_width_height(self):
        self.assertEqual(_chart_content_box(0, 0, 0, 0), (8, 4, -16, -8))


class TestChartPanel(unittest.TestCase):
    """_chart_panel appends a rect and returns the content area."""

    def test_appends_rect_and_returns_content_box(self):
        el: list[str] = []
        result = _chart_panel(el, 100, 200, 500, 220)
        self.assertEqual(len(el), 1)
        self.assertTrue(el[0].startswith("<rect"))
        self.assertEqual(result, (108, 204, 484, 212))

    def test_content_box_matches_standalone_helper(self):
        el: list[str] = []
        result = _chart_panel(el, 16, 500, 1148, 250)
        self.assertEqual(result, _chart_content_box(16, 500, 1148, 250))

    def test_appends_correct_rect(self):
        el: list[str] = []
        _chart_panel(el, 10, 20, 100, 80)
        expected_rect = _chart_panel_rect(10, 20, 100, 80)
        self.assertEqual(el[0], expected_rect)

    def test_panel_stacking(self):
        """Multiple panel calls accumulate in order."""
        el: list[str] = []
        r1 = _chart_panel(el, 0, 0, 100, 50)
        r2 = _chart_panel(el, 0, 0, 100, 50)
        self.assertEqual(len(el), 2)
        self.assertEqual(r1, r2)  # same dims → same content box


class TestChartPanelRect(unittest.TestCase):
    """_chart_panel_rect generates SVG rect elements with standard styling."""

    def test_returns_rect_element(self):
        """Returns a properly formatted SVG rect tag."""
        result = _chart_panel_rect(10, 20, 100, 50)
        self.assertIn("<rect ", result)
        self.assertIn("x='10'", result)
        self.assertIn("y='20'", result)
        self.assertIn("width='100'", result)
        self.assertIn("height='50'", result)

    def test_standard_styling_applied(self):
        """Standard rx, fill, stroke attributes are present."""
        result = _chart_panel_rect(0, 0, 200, 100)
        self.assertIn("rx='6'", result)
        self.assertIn("fill='#fff'", result)
        self.assertIn("stroke='#e2e8f0'", result)
        self.assertIn("stroke-width='1'", result)

    def test_float_coordinates(self):
        """Float coordinates are rendered as-is (no rounding)."""
        result = _chart_panel_rect(10.5, 20.7, 100.3, 50.9)
        self.assertIn("x='10.5'", result)
        self.assertIn("y='20.7'", result)
        self.assertIn("width='100.3'", result)
        self.assertIn("height='50.9'", result)


class TestChartArea(unittest.TestCase):
    """_chart_area computes the chart drawing area from margins."""

    def test_returns_cx_cy_cw_ch(self):
        """Returns (cx, cy, cw, ch) tuple with correct values."""
        cx, cy, cw, ch = _chart_area(10, 20, 200, 100, pl=50, pr=20, pb=40, pt=30)
        self.assertEqual(cx, 60)     # x + pl
        self.assertEqual(cy, 50)     # y + pt
        self.assertEqual(cw, 130)    # w - pl - pr
        self.assertEqual(ch, 30)     # h - pt - pb

    def test_zero_margins(self):
        """When margins are all zero, cx=x, cy=y, cw=w, ch=h."""
        cx, cy, cw, ch = _chart_area(5, 10, 100, 50, pl=0, pr=0, pb=0, pt=0)
        self.assertEqual((cx, cy, cw, ch), (5, 10, 100, 50))

    def test_margins_exceed_dimensions(self):
        """When margins exceed dimensions, cw/ch can become negative."""
        cx, cy, cw, ch = _chart_area(0, 0, 10, 10, pl=20, pr=20, pb=20, pt=20)
        self.assertEqual(cw, -30)    # 10 - 20 - 20
        self.assertEqual(ch, -30)    # 10 - 20 - 20

    def test_float_values(self):
        """All values can be floats."""
        cx, cy, cw, ch = _chart_area(10.5, 20.3, 200.0, 100.0, pl=50.2, pr=20.1, pb=40.0, pt=30.0)
        self.assertAlmostEqual(cx, 60.7)  # 10.5 + 50.2
        self.assertAlmostEqual(cw, 129.7)  # 200 - 50.2 - 20.1


class TestChartEmptyState(unittest.TestCase):
    """_chart_empty_state writes the 'no valid runs' SVG block."""

    def test_appends_header_and_text(self):
        """Appends header and empty-state text to the element list."""
        el: list[str] = []
        _chart_empty_state(el, "Test Title", "Mbps", 10, 20, 200, 100, pl=50, pt=40, ch=60)
        self.assertGreaterEqual(len(el), 2)
        # Should include header and "no valid runs" text
        self.assertTrue(any("Test Title" in s for s in el))
        self.assertTrue(any("no valid runs" in s for s in el))

    def test_does_not_add_data_points(self):
        """Only header and empty-state text — no polyline or circles."""
        el: list[str] = []
        _chart_empty_state(el, "T", "Y", 0, 0, 100, 50, pl=20, pt=10, ch=30)
        # No polyline, no circle, no grid lines
        for s in el:
            self.assertNotIn("polyline", s)
            self.assertNotIn("circle", s)


class TestOrErr(unittest.TestCase):
    """_or_err renders a value or "ERR" — masquerade-trust guard."""

    def test_ok_returns_str_value(self):
        """When is_ok is True, returns str(value)."""
        self.assertEqual(_or_err(42.5, True), "42.5")

    def test_not_ok_returns_err(self):
        """When is_ok is False, returns 'ERR'."""
        self.assertEqual(_or_err(42.5, False), "ERR")

    def test_float_zero(self):
        """Float 0.0 is rendered as '0.0' when ok."""
        self.assertEqual(_or_err(0.0, True), "0.0")

    def test_zero_string(self):
        """String '0' is preserved when ok."""
        self.assertEqual(_or_err("0", True), "0")

    def test_empty_string(self):
        """Empty string is rendered as '' when ok."""
        self.assertEqual(_or_err("", True), "")

    def test_none_when_ok(self):
        """None is rendered as 'None' when ok (caller should filter)."""
        self.assertEqual(_or_err(None, True), "None")

    def test_none_when_not_ok(self):
        """None is rendered as 'ERR' when not ok."""
        self.assertEqual(_or_err(None, False), "ERR")

    def test_bool_false_when_ok(self):
        """bool False is rendered as 'False' when ok."""
        self.assertEqual(_or_err(False, True), "False")


class TestRecordCommand(unittest.TestCase):
    """_record_command records a command result in process["commands"]."""

    def test_records_to_dict_in_commands(self):
        """result.to_dict() is stored under the given label."""
        process: dict[str, Any] = {"commands": {}}
        result = MagicMock()
        result.to_dict.return_value = {"returncode": 0, "stdout": "ok"}

        returned = _record_command(process, "test_cmd", result)

        self.assertIs(returned, result)
        self.assertIn("test_cmd", process["commands"])
        self.assertEqual(process["commands"]["test_cmd"]["returncode"], 0)

    def test_returns_result_unchanged(self):
        """The result object itself is returned (not the dict)."""
        process: dict[str, Any] = {"commands": {}}
        result = MagicMock()

        returned = _record_command(process, "x", result)

        self.assertIs(returned, result)

    def test_multi_record_preserves_order(self):
        """Multiple records accumulate in order."""
        process: dict[str, Any] = {"commands": {}}
        r1 = MagicMock(); r1.to_dict.return_value = {"seq": 1}
        r2 = MagicMock(); r2.to_dict.return_value = {"seq": 2}

        _record_command(process, "first", r1)
        _record_command(process, "second", r2)

        self.assertEqual(len(process["commands"]), 2)
        self.assertEqual(process["commands"]["first"]["seq"], 1)
        self.assertEqual(process["commands"]["second"]["seq"], 2)


class TestRunCleanupCmd(unittest.TestCase):
    """_run_cleanup_cmd runs a remote cleanup and returns evidence."""

    def test_returns_evidence_on_success(self):
        """On .ok=True, returns {evidence_key: to_dict()}."""
        mock_result = MagicMock()
        mock_result.ok = True
        mock_result.to_dict.return_value = {"returncode": 0}

        with patch("rdma_sweep._run_remote_result", return_value=mock_result) as mock_run:
            evidence = _run_cleanup_cmd(
                "some command", "host1", None,
                evidence_key="my_cleanup", error_key="my_error",
            )

        mock_run.assert_called_once_with("some command", "host1", ssh_config=None)
        self.assertEqual(evidence, {"my_cleanup": {"returncode": 0}})
        self.assertNotIn("my_error", evidence)

    def test_returns_evidence_on_failure(self):
        """On .ok=False, includes error_key."""
        mock_result = MagicMock()
        mock_result.ok = False
        mock_result.to_dict.return_value = {"returncode": 1}
        mock_result.error_summary.return_value = "command failed"

        with patch("rdma_sweep._run_remote_result", return_value=mock_result):
            evidence = _run_cleanup_cmd(
                "fail cmd", "host2", {"sudo": True},
                evidence_key="cleanup", error_key="error",
            )

        self.assertEqual(evidence["cleanup"]["returncode"], 1)
        self.assertEqual(evidence["error"], "command failed")

    def test_returns_error_on_exception(self):
        """On exception, returns {error_key: str(exc)}."""
        with patch("rdma_sweep._run_remote_result", side_effect=OSError("connection lost")):
            evidence = _run_cleanup_cmd(
                "cmd", "host3", None,
                evidence_key="ckey", error_key="ekey",
            )

        self.assertNotIn("ckey", evidence)
        self.assertEqual(evidence["ekey"], "connection lost")


class TestComputeNodeDeltas(unittest.TestCase):
    """_compute_node_deltas grabs state and computes CPU/memory deltas for one node."""

    def test_returns_after_cpu_mem(self):
        """Returns (after, cpu_diff, mem_info) tuple."""
        monitor = MagicMock(spec=SysMonitor)
        monitor.grab.return_value = {"idle": 100, "mem": "1024"}
        with patch.object(SysMonitor, "compute_cpu_diff", return_value={"cpu": 25.0}), \
             patch.object(SysMonitor, "extract_mem", return_value={"mem_total": 1024}), \
             patch.object(SysMonitor, "compute_mem_delta", return_value={"mem_delta": 512}):
            after, cpu_diff, mem_info = _compute_node_deltas(monitor, {"idle": 0})

        self.assertEqual(after, {"idle": 100, "mem": "1024"})
        self.assertEqual(cpu_diff, {"cpu": 25.0})
        self.assertEqual(mem_info["mem_total"], 1024)
        self.assertEqual(mem_info["mem_delta"], 512)

    def test_passes_before_to_cpu_diff(self):
        """The 'before' dict is passed to compute_cpu_diff."""
        monitor = MagicMock(spec=SysMonitor)
        monitor.grab.return_value = {"idle": 200}
        before = {"idle": 100}
        with patch.object(SysMonitor, "compute_cpu_diff", return_value={}) as mock_cpu, \
             patch.object(SysMonitor, "extract_mem", return_value={}), \
             patch.object(SysMonitor, "compute_mem_delta", return_value={}):
            _compute_node_deltas(monitor, before)

        mock_cpu.assert_called_once_with(before, {"idle": 200})


class TestGetDict(unittest.TestCase):
    """_get_dict extracts dict values with safe fallback to {}."""

    def test_simple_key(self):
        """Returns the dict value for a present key."""
        d = {"a": {"x": 1}}
        self.assertEqual(_get_dict(d, "a"), {"x": 1})

    def test_missing_key(self):
        """Returns {} for an absent key."""
        self.assertEqual(_get_dict({"a": 1}, "b"), {})

    def test_none_value(self):
        """Returns {} when the value is None."""
        self.assertEqual(_get_dict({"a": None}, "a"), {})

    def test_empty_dict_value(self):
        """Returns {} when the value is an empty dict."""
        self.assertEqual(_get_dict({"a": {}}, "a"), {})

    def test_fallback_key(self):
        """Tries fallback keys when primary key is missing."""
        d = {"b": {"y": 2}}
        self.assertEqual(_get_dict(d, "a", "b"), {"y": 2})

    def test_fallback_key_primary_is_none(self):
        """Falls through when primary key is None."""
        d = {"a": None, "b": {"y": 2}}
        self.assertEqual(_get_dict(d, "a", "b"), {"y": 2})

    def test_all_keys_missing(self):
        """Returns {} when no key is found."""
        self.assertEqual(_get_dict({"a": 1}, "x", "y", "z"), {})

    def test_non_dict_value_skipped(self):
        """Skips a key whose value is a non-dict (e.g. int)."""
        self.assertEqual(_get_dict({"a": 42, "b": {"ok": 1}}, "a", "b"), {"ok": 1})

    def test_does_not_mutate_source(self):
        """The returned dict is a reference to the original (no copy)."""
        inner = {"x": 1}
        result = _get_dict({"a": inner}, "a")
        result["y"] = 2
        self.assertEqual(inner["y"], 2)  # same object

    def test_first_present_dict_wins(self):
        """Primary key takes precedence over fallback."""
        d = {"server_memory": {"used": 100}, "memory": {"used": 50}}
        self.assertEqual(_get_dict(d, "server_memory", "memory"), {"used": 100})

    def test_variable_key(self):
        """Works with a runtime-variable key name."""
        d = {"server_memory": {"used": 100}, "client_memory": {"used": 50}}
        for k, expected in [("server_memory", {"used": 100}), ("client_memory", {"used": 50})]:
            self.assertEqual(_get_dict(d, k), expected)


class TestReadJson(unittest.TestCase):
    """_read_json reads and parses a JSON file."""

    def test_reads_valid_json(self):
        """Returns parsed dict from a valid JSON file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write('{"a": 1, "b": 2}')
            p = Path(f.name)
        try:
            self.assertEqual(_read_json(p), {"a": 1, "b": 2})
        finally:
            p.unlink(missing_ok=True)

    def test_reads_list(self):
        """Returns parsed list from a JSON array file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write('[1, 2, 3]')
            p = Path(f.name)
        try:
            self.assertEqual(_read_json(p), [1, 2, 3])
        finally:
            p.unlink(missing_ok=True)

    def test_raises_on_invalid_json(self):
        """Raises JSONDecodeError for invalid content."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write('not json')
            p = Path(f.name)
        try:
            with self.assertRaises(json.JSONDecodeError):
                _read_json(p)
        finally:
            p.unlink(missing_ok=True)

    def test_raises_on_missing_file(self):
        """Raises FileNotFoundError when file does not exist."""
        with self.assertRaises(FileNotFoundError):
            _read_json(Path("/nonexistent/path.json"))

    def test_empty_file_raises(self):
        """Raises JSONDecodeError on empty file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            p = Path(f.name)
        try:
            with self.assertRaises(json.JSONDecodeError):
                _read_json(p)
        finally:
            p.unlink(missing_ok=True)


class TestWriteJson(unittest.TestCase):
    """_write_json serializes data as JSON and writes to file."""

    def test_writes_valid_json(self):
        """Writes a dict that can be read back correctly."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            p = Path(f.name)
        try:
            data = {"x": 10, "y": 20}
            _write_json(p, data)
            self.assertEqual(json.loads(p.read_text()), data)
        finally:
            p.unlink(missing_ok=True)

    def test_writes_with_indent_default(self):
        """Default indent is 2 spaces."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            p = Path(f.name)
        try:
            _write_json(p, {"a": 1})
            text = p.read_text()
            self.assertIn('  "a"', text)
        finally:
            p.unlink(missing_ok=True)

    def test_writes_with_custom_indent(self):
        """Respects custom indent parameter."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            p = Path(f.name)
        try:
            _write_json(p, {"a": 1}, indent=4)
            text = p.read_text()
            self.assertIn('    "a"', text)
        finally:
            p.unlink(missing_ok=True)

    def test_writes_empty_dict(self):
        """Writes '{}' for an empty dict."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            p = Path(f.name)
        try:
            _write_json(p, {})
            self.assertEqual(p.read_text().strip(), "{}")
        finally:
            p.unlink(missing_ok=True)


class TestFilterPositiveX(unittest.TestCase):
    """_filter_positive_x drops data pairs with non-positive x values."""

    def test_all_positive(self):
        """Returns all pairs when every x > 0."""
        result = _filter_positive_x([1.0, 2.0, 3.0], [10, 20, 30])
        self.assertEqual(result, [(1.0, 10), (2.0, 20), (3.0, 30)])

    def test_filters_zero_x(self):
        """Drops pairs where x == 0."""
        result = _filter_positive_x([0.0, 2.0, 0.0], [10, 20, 30])
        self.assertEqual(result, [(2.0, 20)])

    def test_filters_negative_x(self):
        """Drops pairs where x < 0."""
        result = _filter_positive_x([-1.0, 2.0, -3.0], [10, 20, 30])
        self.assertEqual(result, [(2.0, 20)])

    def test_all_non_positive(self):
        """Returns empty list when no x > 0."""
        self.assertEqual(_filter_positive_x([0, -1, -5], [1, 2, 3]), [])

    def test_empty_input(self):
        """Returns empty list for empty inputs."""
        self.assertEqual(_filter_positive_x([], []), [])

    def test_mixed_types(self):
        """Works with int and float y values."""
        result = _filter_positive_x([1, 2, 3], [0.5, 1.5, 2.5])
        self.assertEqual(result, [(1, 0.5), (2, 1.5), (3, 2.5)])


class TestExtractMetric(unittest.TestCase):
    """_extract_metric extracts metric values from summary with None guard."""

    def test_extracts_values(self):
        """Returns float values for each index in ok_idx."""
        summary = [
            {"BW_average": 1000},
            {"BW_average": 2000},
            {"BW_average": 3000},
        ]
        self.assertEqual(
            _extract_metric(summary, [0, 1, 2], "BW_average"),
            [1000.0, 2000.0, 3000.0],
        )

    def test_filters_by_ok_idx(self):
        """Only extracts values for indices in ok_idx."""
        summary = [
            {"BW_average": 1000},
            {"BW_average": 2000},
            {"BW_average": 3000},
        ]
        self.assertEqual(
            _extract_metric(summary, [0, 2], "BW_average"),
            [1000.0, 3000.0],
        )

    def test_coerces_none_to_zero(self):
        """Converts stored None to 0.0."""
        summary = [
            {"BW_average": 1000},
            {"BW_average": None},
            {"BW_average": 3000},
        ]
        self.assertEqual(
            _extract_metric(summary, [0, 1, 2], "BW_average"),
            [1000.0, 0.0, 3000.0],
        )

    def test_absent_key_uses_default(self):
        """Returns default when key is absent."""
        self.assertEqual(
            _extract_metric([{"other": 42}], [0], "missing"),
            [0.0],
        )

    def test_custom_default(self):
        """Respects custom default value."""
        summary = [{"val": None}]
        self.assertEqual(
            _extract_metric(summary, [0], "val", default=1),
            [1.0],
        )

    def test_empty_ok_idx(self):
        """Returns empty list when ok_idx is empty."""
        self.assertEqual(
            _extract_metric([{"a": 1}], [], "a"),
            [],
        )

    def test_passes_through_zero(self):
        """Genuine 0.0 passes through unchanged."""
        self.assertEqual(
            _extract_metric([{"v": 0}], [0], "v"),
            [0.0],
        )


class TestSvgTitle(unittest.TestCase):
    """_svg_title appends a bold centered chart title SVG element."""

    def test_appends_title_text(self):
        """Appends an SVG text element with the given title."""
        el: list[str] = []
        _svg_title(el, x=10, w=200, y=5, title="Test Chart")
        self.assertEqual(len(el), 1)
        self.assertIn("Test Chart", el[0])
        self.assertIn("x='110.0'", el[0])  # x + w/2 = 110.0
        self.assertIn("y='17'", el[0])     # y + 12 = 17

    def test_has_bold_centered_styling(self):
        """Contains font-weight bold and text-anchor middle attributes."""
        el: list[str] = []
        _svg_title(el, x=0, w=100, y=0, title="X")
        self.assertIn("font-weight='bold'", el[0])
        self.assertIn("text-anchor='middle'", el[0])
        self.assertIn("font-size='14'", el[0])

    def test_html_escaping(self):
        """Special HTML characters in title are escaped."""
        el: list[str] = []
        _svg_title(el, x=0, w=100, y=0, title="A & B < C > D")
        self.assertNotIn("& B < C > D", el[0])
        self.assertIn("A &amp; B &lt; C &gt; D", el[0])

    def test_positioning(self):
        """Applies x + w/2 and y + 12 for coordinates."""
        el: list[str] = []
        _svg_title(el, x=50, w=400, y=20, title="Positioned")
        self.assertIn("x='250.0'", el[0])  # 50 + 400/2 = 250.0
        self.assertIn("y='32'", el[0])     # 20 + 12 = 32


class TestChartLegendItem(unittest.TestCase):
    """_chart_legend_item appends a colored rect + name label SVG legend pair."""

    def test_appends_rect_and_text(self):
        """Appends exactly two SVG elements: a rect and a text."""
        el: list[str] = []
        _chart_legend_item(el, x=100, y=200, color="#dc2626", name="server")
        self.assertEqual(len(el), 2)

    def test_rect_has_swatch_styling(self):
        """The rect is 10x10 with border-radius 2."""
        el: list[str] = []
        _chart_legend_item(el, x=100, y=200, color="#dc2626", name="x")
        self.assertIn("width='10'", el[0])
        self.assertIn("height='10'", el[0])
        self.assertIn("rx='2'", el[0])
        self.assertIn("fill='#dc2626'", el[0])

    def test_text_positioned_relative_to_rect(self):
        """Text x is rect_x + 16, text y is rect_y + 10."""
        el: list[str] = []
        _chart_legend_item(el, x=100, y=200, color="#000", name="legend")
        # rect x='100', text x='116' (100 + 16)
        self.assertIn("x='100'", el[0])
        self.assertIn("x='116'", el[1])
        # rect y='200', text y='210' (200 + 10)
        self.assertIn("y='200'", el[0])
        self.assertIn("y='210'", el[1])

    def test_html_escaping_in_name(self):
        """Special HTML characters in the name are escaped."""
        el: list[str] = []
        _chart_legend_item(el, x=0, y=0, color="#000", name="A & B < C")
        self.assertNotIn("& B < C", el[1])
        self.assertIn("A &amp; B &lt; C", el[1])

    def test_multiple_items_are_independent(self):
        """Each call appends its own pair of SVG elements."""
        el: list[str] = []
        _chart_legend_item(el, x=10, y=10, color="#dc2626", name="a")
        _chart_legend_item(el, x=200, y=300, color="#0891b2", name="b")
        self.assertEqual(len(el), 4)
        self.assertIn("a", el[1])
        self.assertIn("b", el[3])
        self.assertIn("fill='#dc2626'", el[0])
        self.assertIn("fill='#0891b2'", el[2])


class TestChartYLabel(unittest.TestCase):
    """_chart_y_label appends a rotated y-axis label SVG text element."""

    def test_appends_one_element(self):
        """Appends exactly one SVG text element."""
        el: list[str] = []
        _chart_y_label(el, x=10, y=20, pl=50, pt=40, ch=200, label="MB/s")
        self.assertEqual(len(el), 1)

    def test_has_rotate_transform(self):
        """Contains a rotate(-90) transform with centered pivot."""
        el: list[str] = []
        _chart_y_label(el, x=10, y=20, pl=50, pt=40, ch=200, label="X")
        # pivot = (x + pl/2, y + pt + ch/2) = (35, 160)
        self.assertIn("rotate(-90,", el[0])
        self.assertIn("35.0,", el[0])
        self.assertIn("160.0", el[0])

    def test_positioning(self):
        """Uses x + pl/2 and y + pt + ch/2 for text anchor coordinates."""
        el: list[str] = []
        _chart_y_label(el, x=10, y=20, pl=50, pt=40, ch=200, label="MB/s")
        # x' = 10 + 50/2 = 35, y' = 20 + 40 + 200/2 = 160
        self.assertIn("x='35.0'", el[0])
        self.assertIn("y='160.0'", el[0])

    def test_font_styling(self):
        """Has system-ui font, size 11, and #64748b color."""
        el: list[str] = []
        _chart_y_label(el, x=0, y=0, pl=0, pt=0, ch=0, label="X")
        self.assertIn("font-family='system-ui,sans-serif'", el[0])
        self.assertIn("font-size='11'", el[0])
        self.assertIn("fill='#64748b'", el[0])

    def test_html_escaping(self):
        """Special HTML characters in label are escaped."""
        el: list[str] = []
        _chart_y_label(el, x=0, y=0, pl=0, pt=0, ch=0, label="A & B < C")
        self.assertNotIn("& B < C", el[0])
        self.assertIn("A &amp; B &lt; C", el[0])


class TestChartXLabel(unittest.TestCase):
    """_chart_x_label appends a centered x-axis tick label."""

    def test_appends_one_element(self):
        """Appends exactly one element to the list."""
        el: list[str] = []
        _chart_x_label(el, 50, 10, 200, "1024")
        self.assertEqual(len(el), 1)

    def test_has_label_styling(self):
        """Uses font-size=10, fill=#64748b, text-anchor=middle."""
        el: list[str] = []
        _chart_x_label(el, 50, 10, 200, "test")
        self.assertIn("font-size='10'", el[0])
        self.assertIn("fill='#64748b'", el[0])
        self.assertIn("text-anchor='middle'", el[0])

    def test_y_position_18_below_baseline(self):
        """y = cy + ch + 18."""
        el: list[str] = []
        _chart_x_label(el, 50, 10, 200, "x")
        self.assertIn("y='228'", el[0])

    def test_cx_appears(self):
        """cx appears in x attribute."""
        el: list[str] = []
        _chart_x_label(el, 123.5, 10, 200, "x")
        self.assertIn("x='123.5'", el[0])

    def test_text_content(self):
        """The text content appears in the element."""
        el: list[str] = []
        _chart_x_label(el, 50, 10, 200, "64.0")
        self.assertIn("64.0", el[0])

    def test_no_escaping_by_helper(self):
        """Helper does not escape — caller's responsibility."""
        el: list[str] = []
        _chart_x_label(el, 50, 10, 200, "A & B")
        # The raw & passes through because the contract says callers escape
        self.assertIn("A & B", el[0])


class TestChartSysSeries(unittest.TestCase):
    """_chart_sys_series wraps _chart_sys_xy into a named series tuple."""

    def test_returns_name_xs_ys_color_tuple(self):
        """Returns a (name, xs, ys, color) tuple."""
        result = _chart_sys_series("server", [1.0, 2.0], [True, True], [10.0, 20.0], "#dc2626")
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 4)
        name, xs, ys, color = result
        self.assertEqual(name, "server")
        self.assertEqual(xs, [10.0, 20.0])
        self.assertEqual(ys, [1.0, 2.0])
        self.assertEqual(color, "#dc2626")

    def test_filters_by_mask(self):
        """Only items where mask is True are included."""
        result = _chart_sys_series("cpu", [1.0, 2.0, 3.0], [True, False, True], [10.0, 20.0, 30.0], "#000")
        _, xs, ys, _ = result
        self.assertEqual(xs, [10.0, 30.0])
        self.assertEqual(ys, [1.0, 3.0])

    def test_interleaved_mask_preserves_pairing(self):
        """xs and ys stay positionally paired with non-contiguous mask."""
        result = _chart_sys_series("c", [1.0, 2.0, 3.0, 4.0],
                                    [True, False, True, False],
                                    [10.0, 20.0, 30.0, 40.0], "#000")
        _, xs, ys, _ = result
        self.assertEqual(xs, [10.0, 30.0])
        self.assertEqual(ys, [1.0, 3.0])

    def test_all_filtered_out(self):
        """Empty xs and ys when no mask item is True."""
        result = _chart_sys_series("mem", [1.0, 2.0], [False, False], [10.0, 20.0], "#fff")
        _, xs, ys, _ = result
        self.assertEqual(xs, [])
        self.assertEqual(ys, [])

    def test_empty_inputs(self):
        """Empty xs and ys when all input lists are empty."""
        result = _chart_sys_series("x", [], [], [], "#000")
        _, xs, ys, _ = result
        self.assertEqual(xs, [])
        self.assertEqual(ys, [])


class TestDrawYGrid(unittest.TestCase):
    """_draw_y_grid appends 5 grid lines, 5 y-labels, and 1 baseline axis."""

    def setUp(self):
        self.el: list[str] = []

    def test_appends_11_elements(self):
        """5 grid lines + 5 labels + 1 baseline = 11 appended elements."""
        _draw_y_grid(self.el, 10.0, 20.0, 200.0, 100.0, ["a", "b", "c", "d", "e"])
        self.assertEqual(len(self.el), 11)

    def test_grid_lines_have_correct_stroke(self):
        """Horizontal grid lines use #e2e8f0 stroke."""
        _draw_y_grid(self.el, 0.0, 0.0, 100.0, 100.0, ["0", "25", "50", "75", "100"])
        for i in range(5):
            self.assertIn("#e2e8f0", self.el[i * 2])

    def test_labels_routed_correctly(self):
        """Each label text appears in the corresponding text element."""
        labels = ["100", "75", "50", "25", "0"]
        _draw_y_grid(self.el, 0.0, 0.0, 200.0, 100.0, labels)
        for i in range(5):
            self.assertIn(labels[i], self.el[i * 2 + 1])

    def test_baseline_axis_uses_darker_stroke(self):
        """Baseline axis at the bottom uses #94a3b8."""
        _draw_y_grid(self.el, 10.0, 20.0, 200.0, 100.0, ["a", "b", "c", "d", "e"])
        baseline = self.el[-1]
        self.assertIn("#94a3b8", baseline)
        self.assertIn("line", baseline)
        self.assertIn("stroke-width='1'", baseline)

    def test_empty_el_starts_empty(self):
        """Helper: el starts empty before draw_y_grid."""
        self.assertEqual(self.el, [])


class TestDrawSeriesPolyline(unittest.TestCase):
    """_draw_series_polyline renders a polyline + circle markers for a data series."""

    def setUp(self):
        self.el: list[str] = []
        # Geometry: xmn=1, xmx=2 → _log2_px maps 1→0, 2→200
        #           ymn=0, ymx=100 → _scale_y maps 0→200, 100→100
        self.pts = [(1, 0), (2, 100)]
        self.cx, self.cy, self.cw, self.ch = 0.0, 100.0, 200.0, 100.0
        self.xmn, self.xmx, self.ymn, self.ymx = 1.0, 2.0, 0.0, 100.0

    def test_appends_polyline_plus_circles(self):
        """One polyline + N circles = 1 + len(pts) elements."""
        _draw_series_polyline(self.el, self.pts, self.cx, self.cy, self.cw, self.ch,
                              self.xmn, self.xmx, self.ymn, self.ymx, "#dc2626")
        self.assertEqual(len(self.el), 1 + len(self.pts))

    def test_polyline_has_correct_styling(self):
        """Polyline has fill='none', stroke and stroke-width attributes."""
        _draw_series_polyline(self.el, self.pts, self.cx, self.cy, self.cw, self.ch,
                              self.xmn, self.xmx, self.ymn, self.ymx, "#0891b2")
        polyline = self.el[0]
        self.assertIn("polyline", polyline)
        self.assertIn("fill='none'", polyline)
        self.assertIn("stroke='#0891b2'", polyline)
        self.assertIn("stroke-width='2.5'", polyline)

    def test_circles_have_correct_radius(self):
        """Circle elements use the given radius."""
        _draw_series_polyline(self.el, self.pts, self.cx, self.cy, self.cw, self.ch,
                              self.xmn, self.xmx, self.ymn, self.ymx, "#000", r=3.5)
        for i in range(len(self.pts)):
            self.assertIn("r='3.5'", self.el[i + 1])

    def test_circles_have_fill_color(self):
        """Circle elements carry the series color."""
        _draw_series_polyline(self.el, self.pts, self.cx, self.cy, self.cw, self.ch,
                              self.xmn, self.xmx, self.ymn, self.ymx, "#9333ea")
        for i in range(len(self.pts)):
            self.assertIn("fill='#9333ea'", self.el[i + 1])

    def test_coordinates_in_points_string(self):
        """Polyline points string contains computed log2/scaled coordinates."""
        _draw_series_polyline(self.el, self.pts, self.cx, self.cy, self.cw, self.ch,
                              self.xmn, self.xmx, self.ymn, self.ymx, "#000")
        polyline = self.el[0]
        # point 1: log2_px(1)=0, scale_y(0)=200 → "0,200"
        # point 2: log2_px(2)=200, scale_y(100)=100 → "200,100"
        self.assertIn("points='0.0,200.0 200.0,100.0'", polyline)

    def test_three_point_series(self):
        """Works with 3 points, producing 1+3=4 elements."""
        pts3 = [(1, 100), (1.5, 50), (2, 0)]
        _draw_series_polyline(self.el, pts3, self.cx, self.cy, self.cw, self.ch,
                              self.xmn, self.xmx, self.ymn, self.ymx, "#ca8a04")
        self.assertEqual(len(self.el), 4)


class TestSvgEmptyPanel(unittest.TestCase):
    """_svg_empty_panel renders a title + centered placeholder note."""

    def test_appends_two_elements(self):
        """Appends title + placeholder = 2 elements."""
        el: list[str] = []
        _svg_empty_panel(el, 0, 200, 10, "Test Title", 100, "no data")
        self.assertEqual(len(el), 2)

    def test_title_appears(self):
        """First element is the bold title."""
        el: list[str] = []
        _svg_empty_panel(el, 0, 200, 10, "My Title", 100, "empty")
        self.assertIn("My Title", el[0])
        self.assertIn("font-weight='bold'", el[0])

    def test_placeholder_appears(self):
        """Second element is the centered grey placeholder."""
        el: list[str] = []
        _svg_empty_panel(el, 0, 200, 10, "T", 100, "nothing here")
        self.assertIn("nothing here", el[1])
        self.assertIn("fill='#94a3b8'", el[1])

    def test_placeholder_at_given_cy(self):
        """Placeholder uses the supplied cy for y-position."""
        el: list[str] = []
        _svg_empty_panel(el, 0, 200, 10, "T", 123, "x")
        self.assertIn("y='123'", el[1])


class TestSvgPlaceholderText(unittest.TestCase):
    """_svg_placeholder_text appends a centered grey message label."""

    def test_appends_one_element(self):
        """Appends exactly one element to the list."""
        el: list[str] = []
        _svg_placeholder_text(el, 100, 200, "hello")
        self.assertEqual(len(el), 1)

    def test_has_placeholder_styling(self):
        """Uses font-size=12, fill=#94a3b8, text-anchor=middle."""
        el: list[str] = []
        _svg_placeholder_text(el, 100, 200, "test")
        self.assertIn("font-size='12'", el[0])
        self.assertIn("fill='#94a3b8'", el[0])
        self.assertIn("text-anchor='middle'", el[0])

    def test_message_appears(self):
        """The message text is embedded in the element."""
        el: list[str] = []
        _svg_placeholder_text(el, 50, 100, "no data")
        self.assertIn("no data", el[0])

    def test_coordinates_appear(self):
        """Provided cx and cy appear in the element."""
        el: list[str] = []
        _svg_placeholder_text(el, 123.5, 456.7, "x")
        self.assertIn("x='123.5'", el[0])
        self.assertIn("y='456.7'", el[0])

    def test_html_escaping(self):
        """Special chars in msg are HTML-escaped."""
        el: list[str] = []
        _svg_placeholder_text(el, 0, 0, "A & B < C > D")
        self.assertIn("A &amp; B &lt; C &gt; D", el[0])


if __name__ == "__main__":
    unittest.main()
