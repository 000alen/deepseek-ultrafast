import argparse
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

from b200_qualify import benchmark_command, passing, run_case


class QualificationDriverTests(unittest.TestCase):
    def test_cold_kickoff_uses_neither_replenishment_nor_offered_rate_gate(self):
        args = argparse.Namespace(workloads=Path("inputs"), model="test", milestone_s=60, min_cache_samples=20,
                                  warmup_s=10, replenish_s=100, min_offered_rate=8)
        command = benchmark_command(args, "kickoff", 500, "vllm-dep8", "http://127.0.0.1:18080/v1/chat/completions",
                                    Path("inventory.json"))
        self.assertTrue(command[2].endswith("kickoff.json"))
        self.assertEqual(command[command.index("--qualify") + 1], "cold")
        self.assertEqual(command[command.index("--tasks") + 1], "500")
        self.assertEqual(command[command.index("--request-format") + 1], "vllm")
        self.assertNotIn("--replenish-s", command)
        self.assertNotIn("--min-turns-per-s", command)

    def test_fault_marker_invokes_action_once_and_keeps_failed_report(self):
        calls = []

        def fault():
            calls.append(True)
            return {"confirmed_down": True}

        child = ('import sys\n'
                 'for _ in range(2):\n'
                 '    print(\'{"benchmark_event": {"marker_s": 2}}\', file=sys.stderr, flush=True)\n'
                 'print(\'{"qualification": {"passed": false}}\')\n'
                 'sys.exit(1)\n')
        with TemporaryDirectory() as output:
            case = Path(output)
            self.assertEqual(run_case([sys.executable, "-c", child], case, fault), 1)
            receipt = json.loads((case / "fault.json").read_text())
            self.assertEqual(receipt, {"marker": {"marker_s": 2}, "action": {"confirmed_down": True}})
            self.assertTrue((case / "report.json").exists())
        self.assertEqual(calls, [True])

    def test_highest_passing_task_count_ignores_cold_and_failed_larger_cases(self):
        results = [
            {"phase": "closed", "offered_tasks": 100, "runs": [{"passed": True}]},
            {"phase": "closed", "offered_tasks": 250, "runs": [{"passed": False}]},
            {"phase": "kickoff", "offered_tasks": 500, "runs": [{"passed": True}]},
            {"phase": "open", "offered_tasks": 500, "runs": [{"passed": True, "arrival_scale": 1},
                                                            {"passed": False, "arrival_scale": 1.25}]},
        ]
        self.assertEqual(passing(results), {"highest_passing_closed_task_count": 100,
                                           "highest_passing_open_arrival_scale": 1})


if __name__ == "__main__":
    unittest.main()
