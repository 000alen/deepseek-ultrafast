"""CPU-only checks for the native-engine concurrency feasibility probe."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import concurrency_probe as probe


CONFIG = {
    "n_layers": 2,
    "n_mtp_layers": 1,
    "head_dim": 16,
    "index_head_dim": 8,
    "compress_ratios": [2, 0, 1],
    "kv_source_layers": [0],
}


class ConcurrencyProbeTests(unittest.TestCase):
    def test_trace_reader_counts_unmeasurable_requests_without_guessing_shapes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "worker.jsonl"
            trace.write_text("\n".join(json.dumps(row) for row in (
                {"identity": {"run_id": "private"}},
                {"event": "request_end", "session_id": "do-not-print", "input_tokens": 100,
                 "output_tokens": 20, "cached_tokens": 25, "average_iteration_ms": 4.5},
                {"event": "request_end", "input_tokens": None, "output_tokens": None},
                {"event": "request_end", "input_tokens": 300, "output_tokens": 50,
                 "cached_tokens": 0, "average_accept_length": 1.2},
            )) + "\n")
            shapes, omitted, identity_bound = probe.load_shapes(trace)
        self.assertEqual(omitted, 1)
        self.assertTrue(identity_bound)
        self.assertEqual([(shape.resident_tokens, shape.uncached_input_tokens) for shape in shapes],
                         [(120, 75), (350, 300)])

    def test_trace_rejects_mixed_worker_incarnations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "mixed.jsonl"
            trace.write_text("\n".join(json.dumps(row) for row in (
                {"identity": {"run_id": "first"}},
                {"event": "request_end", "input_tokens": 100, "output_tokens": 10},
                {"identity": {"run_id": "second"}},
            )) + "\n")
            with self.assertRaisesRegex(ValueError, "more than one worker"):
                probe.load_shapes(trace)

    def test_explicit_cache_counts_the_current_arrays_per_rank(self) -> None:
        geometry = probe.CacheGeometry(2, 1, 16, 8, (2, 0, 1), frozenset({0}))
        # Three 128-byte windows, one source's compressed data and index,
        # two 256-byte histories, and the 16-byte comparison history.
        self.assertEqual(geometry.explicit_cache_bytes(4), 1648)
        self.assertGreater(geometry.explicit_cache_bytes(8), 1648)

    def test_row_budget_reports_required_step_time_without_extrapolating_it(self) -> None:
        geometry = probe.CacheGeometry(2, 1, 16, 8, (2, 0, 1), frozenset({0}))
        shapes = [probe.Shape(100 + i, 50, 0, 1.5, 4.5) for i in range(4)]
        report = probe.evaluate(shapes, geometry, (4,), max_seq=1024,
                                max_verify_rows=8, draft_tokens=3,
                                accepted_tokens_per_step=1, first_useful_ms=2000,
                                minimum_visible_tps=80, explicit_cache_budget_bytes=None)
        point = report["scenarios"][0]
        self.assertEqual(point["sequences_per_decode_step_at_row_limit"], 2)
        self.assertEqual(point["decode_round_robin_waves"], 2)
        self.assertEqual(point["required_max_decode_step_ms_for_speed_floor_before_prefill"], 6.25)
        self.assertEqual(point["required_visible_output_tokens_per_second"], 320)
        self.assertEqual(report["observed_single_request_median_iteration_ms"], 4.5)
        self.assertIn("not_gpu_performance_prediction", report["kind"])

    def test_over_context_and_graph_overrun_are_rejected(self) -> None:
        geometry = probe.CacheGeometry(2, 1, 16, 8, (2, 0, 1), frozenset({0}))
        shape = [probe.Shape(100, 50, 0, None, None)]
        with self.assertRaisesRegex(ValueError, "max_seq"):
            probe.evaluate(shape, geometry, (1,), max_seq=128, max_verify_rows=8,
                           draft_tokens=3, accepted_tokens_per_step=1,
                           first_useful_ms=2000, minimum_visible_tps=80,
                           explicit_cache_budget_bytes=None)
        with self.assertRaisesRegex(ValueError, "16-row"):
            probe.evaluate(shape, geometry, (1,), max_seq=256, max_verify_rows=17,
                           draft_tokens=3, accepted_tokens_per_step=1,
                           first_useful_ms=2000, minimum_visible_tps=80,
                           explicit_cache_budget_bytes=None)
        with self.assertRaisesRegex(ValueError, "chosen decode row budget"):
            probe.evaluate(shape, geometry, (1,), max_seq=256, max_verify_rows=2,
                           draft_tokens=3, accepted_tokens_per_step=1,
                           first_useful_ms=2000, minimum_visible_tps=80,
                           explicit_cache_budget_bytes=None)

    def test_budget_can_reject_the_copy_per_session_layout(self) -> None:
        geometry = probe.CacheGeometry(2, 1, 16, 8, (2, 0, 1), frozenset({0}))
        shape = [probe.Shape(100, 50, 0, None, None)]
        report = probe.evaluate(shape, geometry, (1,), max_seq=256,
                                max_verify_rows=8, draft_tokens=3,
                                accepted_tokens_per_step=1, first_useful_ms=2000,
                                minimum_visible_tps=80, explicit_cache_budget_bytes=100)
        self.assertTrue(report["scenarios"][0]["copy_cache_exceeds_supplied_budget"])

    def test_cold_and_observed_cache_prefill_are_separate(self) -> None:
        geometry = probe.CacheGeometry(2, 1, 16, 8, (2, 0, 1), frozenset({0}))
        shapes = [probe.Shape(100, 20, 80, None, None), probe.Shape(90, 20, 0, None, None)]
        report = probe.evaluate(shapes, geometry, (2,), max_seq=256,
                                max_verify_rows=8, draft_tokens=3,
                                accepted_tokens_per_step=1, first_useful_ms=2000,
                                minimum_visible_tps=80, explicit_cache_budget_bytes=None)
        point = report["scenarios"][0]
        self.assertEqual(point["synchronized_prompt_tokens_all_cold"], 190)
        self.assertEqual(point["synchronized_uncached_prompt_tokens_observed_cache"], 110)
        self.assertEqual(point["required_prefill_tokens_per_second_all_cold"], 95)
        self.assertEqual(point["required_prefill_tokens_per_second_observed_cache"], 55)

    def test_cli_emits_structured_report_without_request_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.json"
            config.write_text(json.dumps(CONFIG))
            trace = root / "worker.jsonl"
            trace.write_text("\n".join(json.dumps({"event": "request_end", "session_id": "private-session",
                "input_tokens": 100 + i, "output_tokens": 50, "cached_tokens": 0}) for i in range(4)) + "\n")
            result = subprocess.run([sys.executable, str(Path(probe.__file__)), "--trace", str(trace),
                "--config", str(config), "--active", "2,4", "--max-seq", "1024",
                "--draft-tokens", "3"], capture_output=True, text=True, check=True)
        report = json.loads(result.stdout)
        self.assertEqual([point["active_sequences"] for point in report["scenarios"]], [2, 4])
        self.assertEqual(len(report["trace_sha256"]), 64)
        self.assertEqual(len(report["config_sha256"]), 64)
        self.assertFalse(report["single_worker_identity_bound"])
        self.assertNotIn("private-session", result.stdout)


if __name__ == "__main__":
    unittest.main()
