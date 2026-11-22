import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from b200_timing import summarize


def plan(plan_id, **values):
    return {"event": "packed_plan_step", "plan_id": plan_id,
            "wall_ms": 12, "decode_lanes": 32, "decode_ms": 6, "decode_tokens": 32, "decode_window_ms": 10,
            "active_decoder_limit": 32, "busy_prefill_budget_ms": 4, "busy_prefill_rows": 256,
            "prefill_backlog_requests": 2, **values}


class TimingTests(unittest.TestCase):
    def test_reports_anchor_associations_and_restart_boundaries(self):
        blocked = {"busy_prefill_budget_ms": 0, "busy_prefill_rows": 0,
                   "non_prefill_peak_plan_id": 1, "peak_plan_id": 1,
                   "active_decoder_limit": 31}
        anchor = plan(1, retirement_anchor_ms=5, retirement_export_ms=0.4,
                      prefill_encode_rows=256, prefill_ms=4)
        rows = [
            json.dumps(anchor),
            json.dumps({"log": json.dumps(plan(2, **blocked, prefill_encode_rows=1, prefill_ms=0.1))}),
            "2026-11-22T19:00:00Z " + json.dumps(plan(3, **blocked)),
            "diagnostic text containing an untrusted input",
            json.dumps(plan(5, **blocked)),
            json.dumps({"event": "packed_decode_ready"}),
            json.dumps(plan(2, **blocked)),
            json.dumps(plan(3, decode_lanes=0, decode_ms=None, prefill_replay_rows=128,
                            prefill_ms=2, wall_ms=1800)),
        ]
        with TemporaryDirectory() as directory:
            path = Path(directory, "replica.log")
            path.write_text("\n".join(rows))
            report = summarize([path.parent, path])
        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["plans"], 6)
        self.assertEqual(report["sources"][0]["processes"], 2)
        self.assertEqual(report["sources"][0]["plan_gaps"], 1)
        self.assertEqual(report["limiter"]["restricted_plans"], 4)
        self.assertEqual(report["limiter"]["restricted_stretches"], 3)
        self.assertEqual(report["limiter"]["restricted_peak_with_retirement_anchor"], 3)
        self.assertEqual(report["limiter"]["restricted_peak_not_observed"], 1)
        self.assertEqual(report["limiter"]["decode_plans_over_window_with_retirement_anchor"], 1)
        self.assertEqual(report["limiter"]["active_limit_decrease_peak_with_retirement_anchor"], 1)
        self.assertEqual(report["limiter"]["longest_consecutive_restricted_plans"], 2)
        self.assertEqual(report["stage_plans"]["retirement_anchor"], 1)
        self.assertEqual(report["by_decode_lanes"]["32"]["metrics"]["retirement_anchor_ms"]["p99"], 5)
        self.assertEqual(report["by_decode_lanes"]["0"]["metrics"]["prefill_replay_ms"]["p95"], 2)
        self.assertNotIn("wall_ms", report["by_decode_lanes"]["0"]["metrics"])

    def test_empty_invalid_and_limited_logs_are_identified(self):
        with TemporaryDirectory() as directory:
            path = Path(directory, "replica.log")
            invalid = json.dumps({"event": "packed_plan_step", "wall_ms": float("nan")})
            path.write_text(invalid)
            empty = summarize([path])
            path.write_text("\n".join((invalid, json.dumps(plan(1)), json.dumps(plan(2)))))
            limited = summarize([path], max_plans=1)
        self.assertEqual(empty["status"], "no_traces")
        self.assertEqual(empty["invalid_plans"], 1)
        self.assertEqual(limited["status"], "limited")
        self.assertEqual(limited["plans"], 1)
        self.assertEqual(limited["invalid_plans"], 1)


if __name__ == "__main__":
    unittest.main()
