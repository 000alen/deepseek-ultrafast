from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from threading import Event, Thread
from types import SimpleNamespace
import unittest

from b200_serving_smoke import continuation, overlap


class ServingHandler(BaseHTTPRequestHandler):
    def frame(self, *, content=None, reason=None, usage=None):
        choices = [] if usage else [{"delta": {"content": content}, "finish_reason": reason}]
        payload = {"id": "fixture-completion", "choices": choices, "usage": usage}
        self.wfile.write(b"data: " + json.dumps(payload).encode() + b"\n\n")

    def finish_stream(self, prompt=24, cached=0):
        self.frame(reason="stop")
        self.frame(usage={"prompt_tokens": prompt, "completion_tokens": 8,
                          "prompt_tokens_details": {"cached_tokens": cached}})
        self.wfile.write(b"data: [DONE]\n\n")

    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        session = self.headers["X-Session-ID"]
        self.server.requests.append((session, request))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("X-Session-ID", session)
        self.end_headers()
        if session.endswith("-overlap-a"):
            self.frame(content="start A")
            if self.server.b_started.wait(3):
                # The client closes A after B starts. Large writes detect the socket closure.
                try:
                    for _ in range(128):
                        self.frame(content="a" * 32768)
                except (BrokenPipeError, ConnectionResetError):
                    self.server.a_closed.set()
                    return
            self.finish_stream()
        elif session.endswith("-overlap-b"):
            self.frame(content="start B")
            self.server.b_started.set()
            if self.server.a_closed.wait(3):
                self.finish_stream()
        elif session.endswith("-continuation-cold"):
            self.frame(content="cold text")
            self.finish_stream(prompt=1300)
        elif session.endswith("-continuation") and len(request["messages"]) > 1:
            self.frame(content="warm text")
            self.finish_stream(prompt=1300, cached=self.server.warm_cache)
        else:
            self.frame(content="first assistant reply")
            self.finish_stream(prompt=1200)

    def log_message(self, *args):
        pass


class ServingSmokeTests(unittest.TestCase):
    def setUp(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), ServingHandler)
        server.requests = []
        server.b_started = Event()
        server.a_closed = Event()
        server.warm_cache = 1208
        self.server = server
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(thread.join)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        self.args = SimpleNamespace(
            url=f"http://127.0.0.1:{server.server_port}/v1/chat/completions",
            overlap_url=None, continuation_url=None, model="model", api_key=None,
            timeout_s=5, publication_wait_s=0,
        )

    def test_staggered_streams_cancel_and_recover(self):
        report = overlap(self.args, "smoke")
        self.assertTrue(all(report["checks"].values()), report)
        self.assertTrue(self.server.a_closed.is_set())
        self.assertIsNotNone(report["requests"]["a"]["cancel_unix_ms"])

    def test_continuation_reuses_complete_history_and_reports_text_drift(self):
        report = continuation(self.args, "smoke")
        self.assertTrue(all(report["checks"].values()), report)
        first, warm, cold = self.server.requests
        self.assertEqual(first[0], warm[0])
        self.assertNotEqual(first[0], cold[0])
        self.assertEqual(warm[1], cold[1])
        self.assertEqual(warm[1]["messages"][:2], [
            *first[1]["messages"], {"role": "assistant", "content": "first assistant reply"},
        ])
        self.assertFalse(report["warm_and_cold_text_match"])
        self.assertNotIn("first assistant reply", json.dumps(report))

    def test_template_only_cache_hit_cannot_pass(self):
        self.server.warm_cache = 8
        report = continuation(self.args, "smoke")
        self.assertFalse(report["checks"]["same_branch_reused_initial_prompt"])


if __name__ == "__main__":
    unittest.main()
