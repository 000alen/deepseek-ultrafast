from array import array
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from threading import Event, Thread
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from bench_coding import Turn, case_failed, load_sessions, qualification, replay, summarize


def measured_turn(cohort, prompt, cached, *, chunks=(1.1, 1.3, 1.4)):
    return Turn("session-" + cohort, cohort, 1., 1., 2., array("d", chunks),
                output_tokens=256, prompt_tokens=prompt, cached_tokens=cached)


class MetricsTests(unittest.TestCase):
    def test_classification_and_joint_goodput_use_observed_cache_and_stream_stalls(self):
        short_gaps = tuple(1.1 + index / 1000 for index in range(120))
        turns = [
            measured_turn("warm", 1000, 0, chunks=short_gaps + (1.7,)),
            measured_turn("cold", 9000, 8000, chunks=short_gaps),
            measured_turn("unreported", 1000, None, chunks=short_gaps),
            measured_turn("long", 40_000, 0, chunks=short_gaps),
        ]
        args = SimpleNamespace(progress_window_ms=250, target_active=3, mode="open",
                               arrival_scale=1, gpus=8)
        report = summarize(turns, args, 1., 2.)
        self.assertEqual(report["planned_vs_observed_cache_cohorts"]["warm"], {"cold": 1})
        self.assertEqual(report["planned_vs_observed_cache_cohorts"]["cold"], {"warm": 1})
        self.assertEqual(report["reported_uncached_input_tokens"], 42_000)
        self.assertLess(report["inter_chunk_gap_ms"]["p99"], 250)
        self.assertGreater(report["per_turn_max_inter_chunk_gap_ms"]["p99"], 250)
        self.assertEqual(report["turns_with_inter_chunk_stalls"], 1)
        self.assertEqual(report["joint_slo_goodput"]["turns_completed_within_all_slos"], 1)
        self.assertEqual(report["joint_slo_goodput"]["output_tokens_per_gpu_s"], 32)
        self.assertEqual(report["joint_slo_goodput"]["turns_by_failed_criterion"],
                         {"stream_stall": 1, "unreported_cache_usage": 1,
                          "outside_standard_prompt_envelope": 1})
        self.assertFalse(qualification(report, "load")["checks"]["cache_usage_observed"])
        self.assertEqual(report["live_sessions"]["peak"], 4)
        self.assertLess(report["successful_recent_token_progress"]["longest_at_target_s"], 1)

    def test_fixture_warm_label_cannot_substitute_for_cache_hit(self):
        turns = [measured_turn("warm", 1000, 0), measured_turn("cold", 1000, 0)]
        args = SimpleNamespace(progress_window_ms=250, target_active=2, mode="closed",
                               arrival_scale=1, gpus=None)
        report = summarize(turns, args, 1., 2.)
        self.assertFalse(qualification(report, "sessions", 0.5)["checks"]["warm_ttft"])
        self.assertFalse(case_failed(report))
        report["event"] = {"marker_observed": False}
        self.assertTrue(case_failed(report))

    def test_production_requires_evidence_and_explicit_cache_conformance(self):
        cold = [measured_turn("cold", 1000, 0) for _ in range(20)]
        warm = [measured_turn("warm", 9000, 8000) for _ in range(20)]
        args = SimpleNamespace(progress_window_ms=250, target_active=2, mode="open",
                               arrival_scale=1, gpus=8)
        report = summarize(cold + warm, args, 1., 2.)
        baseline = qualification(report, "load")
        self.assertFalse(baseline["checks"]["inventory_gpu_count"])
        self.assertFalse(baseline["checks"]["served_models_observed"])
        self.assertTrue(baseline["checks"]["cache_sample_counts"])
        report.update(replica_inventory={"gpu_count": 8, "replicas": []},
                      served_model_evidence={key: {"status": "observed", "ids": ["candidate"]}
                                             for key in ("before", "after")})
        self.assertTrue(qualification(report, "load")["passed"])
        report["planned_vs_observed_cache_cohorts"]["warm"]["cold"] = 1
        self.assertFalse(qualification(report, "load")["checks"]["explicit_cache_cohorts_agree"])
        self.assertTrue(qualification(report, "load", event_replay=True)["passed"])
        self.assertTrue(qualification(report, "load", post_event_s=.5)["passed"])
        report["measured_cache_cohorts"]["warm"]["successful_turns_with_token_chunks"] -= 1
        self.assertFalse(qualification(report, "load")["checks"]["cache_sample_counts"])
        self.assertTrue(qualification(report, "load", .5)["passed"])

    def test_task_metrics_group_branches_and_distinguish_failure_from_inflight(self):
        main = measured_turn("cold", 1_000, 0)
        main.task = "job"
        branch = measured_turn("warm", 9_000, 8_000)
        branch.task, branch.offered, branch.finished = "job", 2., 3.
        failed = measured_turn("failed", 1_000, 0)
        failed.error = "http_429"
        incomplete = measured_turn("inflight", 1_000, 0)
        incomplete.finished = 4.
        args = SimpleNamespace(progress_window_ms=250, target_active=3, mode="open",
                               arrival_scale=1, gpus=None)
        report = summarize([main, branch, failed, incomplete], args, 1., 3.)
        self.assertEqual(report["live_tasks"], {"peak": 3, "longest_at_target_s": 1.})
        self.assertEqual(report["tasks"]["offered"], 3)
        self.assertEqual(report["tasks"]["completed"], 1)
        self.assertEqual(report["tasks"]["failed"], 1)
        self.assertEqual(report["tasks"]["unfinished_at_window_end"], 1)
        self.assertEqual(report["tasks"]["completion_ms"]["p95"], 2_000)
        self.assertEqual(report["tasks"]["failed_by_error"], {"http_429": 1})


class DependencyTests(unittest.TestCase):
    def test_closed_loop_forks_and_joins_on_branch_completion(self):
        def turn(identifier, dependencies=(), gap=0):
            return {"id": identifier, "depends_on": list(dependencies), "gap_s": gap}

        sessions = [
            {"id": "main", "task_id": "job", "turns": [turn("start", gap=.01), turn("join", ("left", "right"), .02)]},
            {"id": "sub-left", "task_id": "job", "turns": [turn("left", ("start",))]},
            {"id": "sub-right", "task_id": "job", "turns": [turn("right", ("start",))]},
        ]
        right_started = Event()
        completed = {}

        def run(args, session, spec, offered, task):
            name = spec["id"]
            result = Turn(session, "unclassified", offered, time.monotonic(), task=task)
            if name == "left":
                self.assertTrue(right_started.wait(timeout=1))
            if name == "right":
                right_started.set()
            result.finished = time.monotonic()
            completed[name] = result
            return result

        args = SimpleNamespace(mode="closed", workers=2, event_at_s=None)
        with patch("bench_coding.run_turn", side_effect=run):
            epoch, results, _ = replay(args, sessions)
        self.assertEqual(len(results), 4)
        self.assertAlmostEqual(completed["start"].offered - epoch, .01)
        for name in ("left", "right"):
            self.assertEqual(completed[name].offered, completed["start"].finished)
        self.assertEqual(completed["join"].offered, max(completed[name].finished for name in ("left", "right")) + .02)
        self.assertEqual({result.task for result in results}, {"job"})

    def test_replenishment_replaces_finished_task_while_other_task_is_running(self):
        sessions = [
            {"id": "fast", "task_id": "short", "turns": [{"id": "entry", "gap_s": .01}]},
            {"id": "branch", "task_id": "short", "turns": [{"id": "child", "depends_on": ["entry"], "gap_s": .02}]},
            {"id": "slow", "turns": [{"id": "long"}]},
        ]
        observations = {}

        def run(args, session, spec, offered, task):
            result = Turn(session, "unclassified", offered, time.monotonic(), task=task)
            if spec["id"] == "long":
                time.sleep(.25)
            result.finished = time.monotonic()
            observations[task, spec["id"]] = result
            return result

        args = SimpleNamespace(mode="closed", workers=2, event_at_s=None, replenish_s=.15, target_active=2)
        with patch("bench_coding.run_turn", side_effect=run):
            epoch, results, _ = replay(args, sessions)
        entries = sorted((result for (_, name), result in observations.items() if name == "entry"),
                         key=lambda result: result.offered)
        self.assertGreaterEqual(len(entries), 2)
        self.assertAlmostEqual(entries[0].offered - epoch, .01)
        self.assertEqual(entries[1].offered, observations[entries[0].task, "child"].finished)
        for entry in entries:
            child = observations[entry.task, "child"]
            self.assertEqual(child.offered, entry.finished + .02)
            self.assertNotEqual(entry.session, child.session)
        long_calls = sorted((result for (_, name), result in observations.items() if name == "long"),
                            key=lambda result: result.offered)
        self.assertLess(entries[1].started, long_calls[0].finished)
        first_offers = {}
        for result in results:
            first_offers[result.task] = min(first_offers.get(result.task, result.offered), result.offered)
        self.assertLess(max(first_offers.values()), epoch + args.replenish_s)
        self.assertGreater(max(result.finished for result in results), epoch + args.replenish_s)
        self.assertEqual(len({result.session for result in results}), len(results))

    def test_invalid_edges_are_rejected_before_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json"
            root = {"sessions": [
                {"id": "one", "turns": [{"id": "a", "depends_on": ["b"], "body": {}}]},
                {"id": "two", "turns": [{"id": "b", "depends_on": ["a"], "body": {}}]},
            ]}
            path.write_text(json.dumps(root))
            with self.assertRaisesRegex(ValueError, "same task"):
                load_sessions(path, "closed", None)
            for session in root["sessions"]:
                session["task_id"] = "job"
            path.write_text(json.dumps(root))
            with self.assertRaisesRegex(ValueError, "cycle"):
                load_sessions(path, "closed", None)


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.requests = []
        self.probes = []
        requests = self.requests
        probes = self.probes

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                probes.append((self.path, self.headers.get("Authorization")))
                if self.path == "/v1/models":
                    payload = {"data": [{"id": "candidate"}, {"id": "vllm"}]}
                elif self.path == "/internal/admission":
                    payload = {"epoch": "mock-epoch"}
                elif self.path == "/readyz":
                    payload = {}
                else:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(payload).encode())

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                requests.append((self.path, self.headers["X-Session-ID"],
                                 self.headers["X-Dynamo-Session-ID"], body))
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                if self.path == "/truncated":
                    self.send_header("Transfer-Encoding", "chunked")
                    self.end_headers()
                    self.wfile.write(b"A\r\ndata:")
                    return
                self.end_headers()

                def send(payload):
                    self.wfile.write(b"data: " + json.dumps(payload).encode() + b"\n\n")
                    self.wfile.flush()

                send({"choices": [{"delta": {"role": "assistant"}}]})
                send({"choices": [{"delta": {"reasoning_content": "plan"}}]})
                time.sleep(0.003)
                send({"choices": [{"delta": {"tool_calls": [{"function": {"arguments": "{}"}}]}}]})
                if self.path == "/event":
                    time.sleep(0.3)
                send({"choices": [{"delta": {"content": "response"}}]})
                send({"choices": [{"delta": {}, "finish_reason": "stop"}]})
                cached = body["messages"][0]["content"] == "warm"
                send({"choices": [], "usage": {"completion_tokens": 16 if self.path == "/short" else 256,
                      "prompt_tokens": 2048,
                      "prompt_tokens_details": {"cached_tokens": 1024 if cached else 0}}})
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.thread.join)
        self.addCleanup(self.server.shutdown)
        self.workspace = tempfile.TemporaryDirectory()
        self.addCleanup(self.workspace.cleanup)
        self.manifest = Path(self.workspace.name) / "trace.json"
        self.manifest.write_text(json.dumps({"sessions": [{"id": "coding", "turns": [
            {"cohort": kind, "at_s": 0.01 * idx, "body": {"model": "candidate",
             "max_completion_tokens": 512, "messages": [{"role": "user", "content": kind}],
             "thinking": {"type": "disabled"}, "reasoning_effort": 75}}
            for idx, kind in enumerate(("cold", "warm"))
        ]}]}))
        self.base = "http://127.0.0.1:" + str(self.server.server_port)

    def benchmark_process(self, path, *args, env=None):
        return subprocess.run(
            [sys.executable, str(Path(__file__).with_name("bench_coding.py")), str(self.manifest),
             "--url", self.base + path, "--api-key-env", "DSUF_BENCH_TEST_ONLY_API_KEY", *args],
            capture_output=True, text=True, timeout=5, env=env,
        )

    def run_benchmark(self, path, *args, env=None):
        process = self.benchmark_process(path, *args, env=env)
        self.assertEqual(process.returncode, 0, process.stderr + "\n" + process.stdout)
        return json.loads(process.stdout), process.stderr

    def test_open_sweep_replays_same_workload_against_both_backends(self):
        report, _ = self.run_benchmark("/candidate", "--mode", "open", "--arrival-scales", "1,1.5",
                                       "--qualify", "load", "--milestone-s", ".001",
                                       "--baseline-url", self.base + "/baseline",
                                       "--baseline-model", "vllm", "--baseline-format", "vllm",
                                       "--baseline-session-header", "X-Dynamo-Session-ID",
                                       "--session-prefix", "trial:", "--isolate-replays",
                                       "--gpus", "8", "--baseline-gpus", "8")
        self.assertEqual(len(report["runs"]), 4)
        self.assertEqual(report["highest_passing_arrival_scale"], {"candidate": 1.5, "baseline": 1.5})
        self.assertEqual(report["workload"]["turns"], 2)
        self.assertEqual(report["workload"]["explicit_output_cap_tokens"]["p95"], 512)
        for run in report["runs"]:
            self.assertEqual(run["workload"]["manifest_sha256"], report["workload"]["manifest_sha256"])
            self.assertEqual(run["joint_slo_goodput"]["turns_completed_within_all_slos"], 2)
            self.assertTrue(run["qualification"]["passed"])
        native_sessions = set()
        vllm_salts = set()
        for path, native_session, dynamo_session, body in self.requests:
            session = dynamo_session if path == "/baseline" else native_session
            self.assertTrue(session.startswith("trial:") and session.endswith(":coding"))
            self.assertTrue(body["stream_options"]["include_usage"])
            if path == "/baseline":
                self.assertIsNone(native_session)
                self.assertEqual(body["model"], "vllm")
                self.assertEqual(body["chat_template_kwargs"],
                                 {"thinking_mode": "chat", "enable_thinking": False, "reasoning_effort": 75})
                self.assertNotIn("thinking", body)
                self.assertNotIn("reasoning_effort", body)
                vllm_salts.add(body["cache_salt"])
            else:
                self.assertIsNone(dynamo_session)
                self.assertEqual(body["thinking"], {"type": "disabled"})
                native_sessions.add(session)
        self.assertEqual(len(native_sessions), 2)
        self.assertEqual(len(vllm_salts), 2)

    def test_operator_event_marks_and_slices_replay_without_injecting_failure(self):
        report, stderr = self.run_benchmark("/event", "--event-at-s", "0.05", "--post-event-s", "0.1",
                                            "--event-label", "host-loss")
        marker = json.loads(stderr)["benchmark_event"]
        self.assertEqual(marker["label"], "host-loss")
        event = report["event"]
        self.assertTrue(event["marker_observed"])
        self.assertTrue(event["full_window_observed"])
        self.assertFalse(event["external_action_verified"])
        self.assertEqual(event["post_event"]["wall_s"], 0.1)
        self.assertEqual(len(self.requests), 2)

    def test_truncated_http_stream_counts_as_failed_turn(self):
        report, _ = self.run_benchmark("/truncated")
        self.assertEqual(report["turns_offered"], 2)
        self.assertEqual(report["turns_completed"], 0)
        self.assertEqual(report["errors"], {"transport_error": 2})

    def test_task_limit_keeps_every_branch_and_qualifies_task_occupancy(self):
        cold, warm = json.loads(self.manifest.read_text())["sessions"][0]["turns"]
        cold["id"] = "entry"
        warm["depends_on"] = ["entry"]
        self.manifest.write_text(json.dumps({"sessions": [
            {"id": "main", "task_id": "job", "turns": [cold]},
            {"id": "excluded", "turns": [{**cold, "id": "other"}]},
            {"id": "branch", "task_id": "job", "turns": [warm]},
        ]}))
        report, _ = self.run_benchmark("/candidate", "--tasks", "1", "--qualify", "tasks",
                                       "--milestone-s", "0.001")
        self.assertEqual([session for _, session, _, _ in self.requests], ["main", "branch"])
        self.assertEqual(report["workload"]["tasks"], 1)
        self.assertEqual(report["workload"]["sessions"], 2)
        self.assertEqual(report["target_active"], 1)
        self.assertEqual(report["tasks"]["completed"], 1)
        self.assertTrue(report["qualification"]["checks"]["sustained_concurrency"])

    def test_cold_only_qualification_accepts_one_short_cold_turn(self):
        root = json.loads(self.manifest.read_text())
        root["sessions"][0]["turns"] = root["sessions"][0]["turns"][:1]
        self.manifest.write_text(json.dumps(root))
        report, _ = self.run_benchmark("/short", "--mode", "open", "--qualify", "cold",
                                       "--milestone-s", "600")
        gates = report["qualification"]
        self.assertTrue(gates["passed"])
        self.assertEqual(gates["cache_samples"], {"cold": 1})
        self.assertIsNone(gates["required_sustained_s"])
        self.assertNotIn("decode_rate", gates["checks"])

    def test_replenishment_reports_fresh_task_instances_and_cache_namespaces(self):
        report, _ = self.run_benchmark("/candidate", "--tasks", "1", "--replenish-s", ".1",
                                       "--request-format", "vllm", "--qualify", "tasks", "--milestone-s", ".01")
        replenished = report["replenishment"]
        self.assertEqual(replenished["source"], "rotating_frozen_task_corpus")
        self.assertEqual(replenished["duration_s"], .1)
        self.assertEqual(replenished["target_active_tasks"], 1)
        self.assertGreater(replenished["task_instances"], 1)
        self.assertEqual(report["tasks"]["completed"], replenished["task_instances"])
        self.assertEqual(report["live_tasks"]["peak"], 1)
        self.assertTrue(report["isolated_cache_namespace"])
        seen_salts = {}
        for _, session, _, body in self.requests:
            salt = body["cache_salt"]
            self.assertEqual(seen_salts.setdefault(session, salt), salt)
        self.assertEqual(len(seen_salts), replenished["task_instances"])
        self.assertEqual(len(set(seen_salts.values())), replenished["task_instances"])

    def inventory_file(self):
        inventory = Path(self.workspace.name) / "inventory.json"
        inventory.write_text(json.dumps({
            "schema_version": 1,
            "endpoint": self.base + "/v1/chat/completions",
            "replicas": [{
                "id": "standard-0",
                "host": "mock-node",
                "gpu_uuids": ["GPU-00000000-0000-0000-0000-000000000001"],
                "epoch": "mock-epoch",
                "decode_width": 32,
                "topology": {"backend": "native", "stage": "combined", "tp": 1, "dp": 1, "ep": 1},
                "startup": {"source": "packed_decode_ready", "ready": True, "image_digest": "sha256:" + "a"*64,
                            "checkpoint_revision": "b"*40, "max_decode_width": 32},
                "admission_url": self.base + "/internal/admission",
                "readiness_url": self.base + "/readyz",
            }],
        }))
        return str(inventory)

    def test_inventory_endpoint_and_models_are_observed_without_credential_disclosure(self):
        token = "private-bench-token"
        env = {**os.environ, "DSUF_BENCH_TEST_ONLY_API_KEY": token}
        path = "/v1/chat/completions?api_key=private-query#private-fragment"
        report, _ = self.run_benchmark(path, "--inventory", self.inventory_file(), "--gpus", "1", env=env)
        self.assertEqual(report["endpoint"], {"origin": self.base, "path": "/v1/chat/completions"})
        self.assertEqual(report["served_model_evidence"]["before"]["ids"], ["candidate", "vllm"])
        self.assertEqual(report["served_model_evidence"]["after"]["ids"], ["candidate", "vllm"])
        inventory = report["replica_inventory"]
        self.assertEqual(inventory["gpu_count"], 1)
        self.assertRegex(inventory["sha256"], r"^[0-9a-f]{64}$")
        replica = inventory["replicas"][0]
        self.assertEqual(replica["provenance"], ["pinned_operator_inventory", "live_readiness_observed",
                                                 "live_admission_epoch_matched"])
        self.assertEqual(replica["live_observations"]["after"]["admission_epoch"]["status"], "matched")
        self.assertIn(("/v1/models", "Bearer " + token), self.probes)
        self.assertIn(("/internal/admission", None), self.probes)
        self.assertIn(("/readyz", None), self.probes)
        self.assertNotIn(token, json.dumps(report))
        self.assertNotIn("private-query", json.dumps(report))
        self.assertNotIn("private-fragment", json.dumps(report))

    def test_production_preflight_rejects_missing_or_mismatched_inventory(self):
        path = "/v1/chat/completions"
        missing = self.benchmark_process(path, "--qualify", "load", "--gpus", "1")
        self.assertEqual(missing.returncode, 2)
        self.assertIn("production qualification requires --gpus and --inventory", missing.stderr)
        mismatch = self.benchmark_process(path, "--qualify", "load", "--gpus", "2",
                                          "--inventory", self.inventory_file())
        self.assertEqual(mismatch.returncode, 2)
        self.assertIn("GPU count does not match", mismatch.stderr)
        self.assertEqual(self.requests, [])
        self.assertEqual(self.probes, [])


if __name__ == "__main__":
    unittest.main()
