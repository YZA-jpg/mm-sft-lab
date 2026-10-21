"""Durable v2 SSE and stage resume with synthetic events, no real network/GPU."""

import contextlib
import io
import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation_v2.common import read_journal, read_json, write_json
from evaluation_v2.config import validate
from evaluation_v2.demo import make_demo
from evaluation_v2.runner import initialize, stage
from evaluation_v2.transport import Client
from evaluation_v2.status import status


def sse(text="Synthetic answer", *, done=True, finish="stop"):
    chunks = [{"id": "synthetic", "model": "target", "choices": [
        {"index": 0, "delta": {"content": text}, "finish_reason": None}],
        "usage": {"completion_tokens": 3}}]
    if finish:
        chunks.append({"id": "synthetic", "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]})
    data = "".join("data: " + json.dumps(chunk, ensure_ascii=False) + "\n\n" for chunk in chunks)
    return io.BytesIO((data + ("data: [DONE]\n\n" if done else "")).encode())


class StreamingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.config = validate({"benchmarks": ["mmstar"], "execution": {
            "generation_stream": True, "generation_concurrency": 1, "per_benchmark_limit": 1}})
        self.endpoint = {"model": "target", "base_url": "http://127.0.0.1:18004/v1", "api_key_env": None}

    def client(self, opener):
        return Client(self.endpoint, self.config["execution"], opener=opener, sleeper=lambda _: None)

    def test_partial_capture_and_usage_are_preserved_before_a_successful_retry(self):
        calls = []
        def opener(request, **kwargs):
            body = json.loads(request.data)
            self.assertEqual(set(body), {"model", "messages", "stream", "stream_options"})
            calls.append(body)
            return sse("partial", done=False, finish=None) if len(calls) == 1 else sse("完整答案")
        result = self.client(opener).complete(messages=[], stream=True, stream_directory=self.root / "capture")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["text"], "完整答案")
        self.assertEqual(result["earlier_partial_attempts"], 1)
        self.assertEqual(result["attempts"][0]["exception_class"], "IncompleteRead")
        self.assertEqual(result["attempts"][0]["http_status"], 200)
        self.assertEqual((self.root / "capture/attempt-01/response.txt").read_text(), "partial")
        self.assertEqual(result["usage"]["completion_tokens"], 3)
        self.assertTrue(result["stream_captures"][0]["partial_output"])

    def test_format_errors_guardrails_and_manual_interrupt_do_not_retry(self):
        malformed = self.client(lambda *args, **kwargs: io.BytesIO(b'data: invalid-json\n\n')).complete(messages=[], stream=True)
        self.assertEqual(len(malformed["attempts"]), 1)
        self.assertEqual(malformed["error_kind"], "api_protocol_error")
        guard = self.client(lambda *args, **kwargs: sse("", finish="content_filter")).complete(messages=[], stream=True)
        self.assertEqual(len(guard["attempts"]), 1)
        self.assertEqual(guard["error_kind"], "provider_guardrail")
        class Interrupted(io.BytesIO):
            def __iter__(self):
                yield b'data: {"choices":[{"delta":{"content":"partial"}}]}\n'
                yield b'\n'
                raise KeyboardInterrupt
        with self.assertRaises(KeyboardInterrupt):
            self.client(lambda *args, **kwargs: Interrupted()).complete(messages=[], stream=True, stream_directory=self.root / "manual")
        self.assertEqual(read_json(self.root / "manual/attempt-01/summary.json")["status"], "interrupted")

    def test_stage_saves_stream_identity_and_resume_skips_success(self):
        data, run = self.root / "data", self.root / "run"
        make_demo(data)
        initialize(data, run, self.config, {"kind": "mock", "label": "synthetic"})
        calls = []
        def factory(endpoint, execution):
            def opener(request, **kwargs):
                calls.append(json.loads(request.data))
                return sse("A")
            return Client(endpoint, execution, opener=opener)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(stage(run, "generation", client_factory=factory)["scoring_complete"])
            stage(run, "generation", client_factory=factory)
        self.assertEqual(len(calls), 1)
        row, = read_journal(run / "generation/results.jsonl")
        self.assertTrue(row["stream"])
        path = Path(row["stream_captures"][0]["directory"])
        self.assertEqual((path / "response.txt").read_text(), "A")
        self.assertEqual(read_json(path.parent / "identity.json")["job_id"], row["job_id"])

    def test_stream_configuration_is_bound_and_no_non_stream_option_leaks(self):
        with self.assertRaises(ValueError):
            validate({"execution": {"generation_stream": "true"}})
        calls = []
        def opener(request, **kwargs):
            calls.append(json.loads(request.data))
            return io.BytesIO(b'{"choices":[{"message":{"content":"answer"},"finish_reason":"stop"}]}')
        self.client(opener).complete(messages=[])
        self.assertEqual(set(calls[0]), {"model", "messages"})

    def test_stream_retries_keep_budget_after_interruption_and_never_overwrite(self):
        def opener(*args, **kwargs):
            raise TimeoutError("synthetic timeout")
        first = self.client(opener).complete(messages=[], stream=True, stream_directory=self.root / "offset", attempt_offset=4)
        self.assertEqual([r["number"] for r in first["attempts"]], [5, 6])
        before = (self.root / "offset/attempt-05/summary.json").read_bytes()
        with self.assertRaises(FileExistsError):
            self.client(opener).complete(messages=[], stream=True, stream_directory=self.root / "offset", attempt_offset=4)
        self.assertEqual((self.root / "offset/attempt-05/summary.json").read_bytes(), before)

    def test_cooperative_cancellation_stops_an_active_stream_and_saves_partial_text(self):
        cancelled = threading.Event()
        class Active(io.BytesIO):
            def __iter__(self):
                yield b'data: {"choices":[{"delta":{"content":"partial"}}]}\n'
                yield b'\n'
                cancelled.set()
                yield b'data: {"choices":[{"delta":{"content":"never retained"}}]}\n'
        with self.assertRaises(KeyboardInterrupt):
            self.client(lambda *args, **kwargs: Active()).complete(messages=[], stream=True,
                stream_directory=self.root / "cooperative", cancel_event=cancelled)
        folder = self.root / "cooperative/attempt-01"
        self.assertEqual((folder / "response.txt").read_text(), "partial")
        self.assertEqual(read_json(folder / "summary.json")["status"], "interrupted")

    def test_live_status_reports_incomplete_stream_progress_without_counting_a_success(self):
        data, run = self.root / "data", self.root / "run"
        make_demo(data)
        manifest = initialize(data, run, self.config, {"kind": "mock"})
        capture = run / "generation/streams/synthetic/attempt-01"
        write_json(capture.parent / "identity.json", {"job_id": "mmstar/test/a#g0", "run_fingerprint": manifest["run_fingerprint"]})
        write_json(capture / "progress.json", {"updated_at": "synthetic-time", "completion_tokens_reported": 1234})
        report = status(run)
        self.assertEqual(report["incomplete_streams"][0]["completion_tokens_reported"], 1234)
        self.assertEqual(report["phases"]["generation"]["valid"], 0)
        write_json(capture / "summary.json", {"status": "ok"})
        self.assertEqual(status(run)["incomplete_streams"], [])


if __name__ == "__main__":
    unittest.main()
