"""Tests for rdma_sweep — unit-testable components.

Run: python3 -m pytest rdma_sweep_tool/test_rdma_sweep.py -v
Or:  python3 -m rdma_sweep_tool.test_rdma_sweep
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from rdma_sweep import _parse_perf_report, _parse_json_output, SysMonitor


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
