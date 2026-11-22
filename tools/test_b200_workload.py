from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from b200_workload import (
    MAX_CONTEXT, PARITY_DECODE, parity_prompts, prepare_replay, synthetic_trace, validate_standard_trace,
)
from bench_coding import load_sessions


class WorkloadTests(unittest.TestCase):
    def test_captured_mixed_keeps_exact_bodies_timing_and_dependencies(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "trace.json"
            sessions = synthetic_trace(11)
            for session in sessions:
                del session["source"]
            sessions[0]["turns"][0]["prompt_tokens"] = 1200
            source.write_text(json.dumps({"sessions": sessions, "trace_kind": "captured"}, indent=2))
            output = root / "prepared"
            args = SimpleNamespace(command="prepare", trace=source, output=output, kickoff_sessions=500,
                                   max_seq=MAX_CONTEXT)
            with redirect_stdout(StringIO()):
                prepare_replay(args)
            self.assertEqual((output / "mixed.json").read_bytes(), source.read_bytes())
            kickoff = load_sessions(output / "kickoff.json", "open", None)
            self.assertEqual(len(kickoff), 11)
            self.assertEqual(kickoff[0]["turns"][0]["body"], sessions[0]["turns"][0]["body"])
            for session in kickoff:
                turn, = session["turns"]
                self.assertEqual(turn["at_s"], 0)
                self.assertNotIn("depends_on", turn)
                self.assertTrue(session["id"].endswith("-main"))
            report = json.loads((output / "workload.json").read_text())
            self.assertEqual(report["source"], "captured")
            self.assertEqual(report["tasks"], 11)
            self.assertEqual(report["branches"], 13)
            self.assertEqual(report["requested_kickoff_requests"], 500)
            self.assertEqual(report["calls_without_trace_prompt_token_count"], 23)

    def test_smoke_is_explicitly_synthetic_and_replays_in_both_modes(self):
        with TemporaryDirectory() as temporary:
            output = Path(temporary) / "smoke"
            args = SimpleNamespace(command="smoke", tasks=3, output=output, max_seq=MAX_CONTEXT)
            with redirect_stdout(StringIO()):
                prepare_replay(args)
            for mode in ("closed", "open"):
                sessions = load_sessions(output / "mixed.json", mode, None)
                self.assertEqual(sessions, synthetic_trace(3))
            self.assertEqual(json.loads((output / "workload.json").read_text())["source"], "synthetic_smoke")
            args = SimpleNamespace(command="prepare", trace=output / "mixed.json", kickoff_sessions=3,
                                   output=output.parent / "relabel", max_seq=MAX_CONTEXT)
            with self.assertRaisesRegex(ValueError, "synthetic smoke fixtures"):
                prepare_replay(args)
            self.assertFalse(args.output.exists())

    def test_explicit_output_and_token_evidence_enforce_standard_budget(self):
        sessions = synthetic_trace(1)
        first = sessions[0]["turns"][0]
        first["prompt_tokens"] = MAX_CONTEXT - 256 - 64
        validate_standard_trace(sessions, MAX_CONTEXT)
        first["prompt_tokens"] += 1
        with self.assertRaisesRegex(ValueError, "exceed --max-seq"):
            validate_standard_trace(sessions, MAX_CONTEXT)
        del first["body"]["max_completion_tokens"]
        with self.assertRaisesRegex(ValueError, "explicit positive output cap"):
            validate_standard_trace(sessions, MAX_CONTEXT)

    def test_parity_fixture_fits_reserve_and_crosses_ring_boundary(self):
        class Tokenizer:
            @staticmethod
            def encode(text, *, add_special_tokens):
                assert not add_special_tokens
                return SimpleNamespace(ids=[ord(char) for char in text])

        prompts = parity_prompts(Tokenizer(), MAX_CONTEXT)
        lengths = set(map(len, prompts))
        self.assertEqual(len(prompts), 32)
        self.assertTrue({127, 128, 129, 8318, 8319, 8320}.issubset(lengths))
        self.assertEqual(max(lengths) + PARITY_DECODE, MAX_CONTEXT)
        self.assertTrue(all(len(prompt) + PARITY_DECODE <= 8192 for prompt in parity_prompts(Tokenizer(), 8192)))


if __name__ == "__main__":
    unittest.main()
