"""Streaming diagnostics: synthetic SSE only, no GPU or network connections."""

import contextlib
import io
import json
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
import probe_vllm_stream as probe
import test_vllm_scripts as fixtures


def sse(text="Synthetic answer", finish="stop", done=True):
    chunks = [
        {"id": "synthetic-id", "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}}]},
        {"id": "synthetic-id", "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
         "usage": {"prompt_tokens": 10, "completion_tokens": 3}},
    ]
    if finish:
        chunks.append({"id": "synthetic-id", "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]})
        chunks.append({"id": "synthetic-id", "choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 3}})
    data = "".join("data: " + json.dumps(chunk, ensure_ascii=False) + "\n\n" for chunk in chunks)
    return io.BytesIO((data + ("data: [DONE]\n\n" if done else "")).encode())


class StreamTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name)
        self.payload = probe.streaming_payload({"model": "target", "messages": []})

    def capture(self, opener, name="run"):
        with contextlib.redirect_stdout(io.StringIO()):
            return probe.capture_with_retries("http://127.0.0.1:18004/v1", self.payload,
                self.folder / name, 600, 10, opener=opener, sleeper=lambda _: None)

    def test_complete_stream_assembles_text_and_final_usage_without_decoding_fields(self):
        calls = []
        def opener(request, timeout):
            calls.append(json.loads(request.data))
            self.assertEqual(timeout, 600)
            return sse("合成测试答案")
        result = self.capture(opener)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(set(calls[0]), {"model", "messages", "stream", "stream_options"})
        self.assertEqual(calls[0]["stream_options"], {"include_usage": True, "continuous_usage_stats": True})
        self.assertEqual((self.folder / "run/attempt-01/response.txt").read_text(encoding="utf-8"), "合成测试答案")
        self.assertEqual(result["attempts"][0]["usage"]["completion_tokens"], 3)
        self.assertTrue(result["attempts"][0]["done_received"])
        with self.assertRaises(ValueError):
            probe.streaming_payload({"model": "target", "messages": [], "max_tokens": 100})

    def test_continuously_active_stream_is_not_cut_off_at_total_600_seconds(self):
        clock = iter(range(0, 100_000, 200))
        with patch.object(probe.time, "perf_counter", side_effect=lambda: next(clock)):
            result = self.capture(lambda *args, **kwargs: sse())
        self.assertEqual(result["status"], "ok")
        self.assertGreater(result["attempts"][0]["wall_seconds"], 600)

    def test_partial_network_attempt_is_retained_separately_from_successful_retry(self):
        calls = []
        def opener(*args, **kwargs):
            calls.append(1)
            return sse("partial first answer", finish=None, done=False) if len(calls) == 1 else sse("second answer")
        result = self.capture(opener)
        self.assertEqual(result["successful_attempt"], 2)
        self.assertEqual(result["earlier_partial_attempts"], 1)
        self.assertEqual(result["attempts"][0]["exception_class"], "IncompleteRead")
        self.assertEqual((self.folder / "run/attempt-01/response.txt").read_text(), "partial first answer")
        self.assertEqual((self.folder / "run/attempt-02/response.txt").read_text(), "second answer")

    def test_guardrails_and_format_errors_are_not_retried(self):
        def guard(*args, **kwargs):
            raise urllib.error.HTTPError("url", 400, "guard", {}, io.BytesIO(b'{"code":"data_inspection_failed"}'))
        result = self.capture(guard, "guard")
        self.assertEqual(len(result["attempts"]), 1)
        self.assertEqual(result["attempts"][0]["error_kind"], "provider_guardrail")
        bad = self.capture(lambda *args, **kwargs: io.BytesIO(b"data: not-json\n\n"), "bad")
        self.assertEqual(len(bad["attempts"]), 1)
        self.assertEqual(bad["attempts"][0]["error_kind"], "format_error")
        filtered = self.capture(lambda *args, **kwargs: sse("", finish="content_filter"), "filtered")
        self.assertEqual(len(filtered["attempts"]), 1)
        self.assertEqual(filtered["attempts"][0]["error_kind"], "provider_guardrail")

    def test_network_budget_is_initial_attempt_plus_five_retries(self):
        def opener(*args, **kwargs):
            raise TimeoutError("synthetic socket inactivity")
        result = self.capture(opener)
        self.assertEqual(result["status"], "error")
        self.assertEqual(len(result["attempts"]), 6)
        self.assertTrue(all(r["exception_class"] == "TimeoutError" for r in result["attempts"]))

    def test_manual_interrupt_preserves_partial_output_and_is_never_success_or_retry(self):
        class Interrupted(io.BytesIO):
            def __iter__(self):
                yield b'data: {"id":"test","choices":[{"index":0,"delta":{"content":"retained partial"}}]}\n'
                yield b'\n'
                raise KeyboardInterrupt
        result = self.capture(lambda *args, **kwargs: Interrupted())
        self.assertEqual(result["status"], "interrupted")
        self.assertEqual(len(result["attempts"]), 1)
        self.assertTrue(result["attempts"][0]["partial_output"])
        self.assertEqual((self.folder / "run/attempt-01/response.txt").read_text(), "retained partial")

    def test_cli_loads_existing_prepared_contract_and_writes_complete_artifacts(self):
        fixture = fixtures.ScriptTests()
        fixture.setUp()
        self.addCleanup(fixture.tearDown)
        output = self.folder / "cli"
        capture = probe.capture_with_retries
        def wrapper(*args, **kwargs):
            return capture(*args, **kwargs, opener=lambda *a, **kw: sse(), sleeper=lambda _: None)
        with patch.object(probe, "wait_for_server", return_value=0), \
             patch.object(probe, "capture_with_retries", side_effect=wrapper), contextlib.redirect_stdout(io.StringIO()):
            result = probe.main(["--server-dir", str(fixture.server), "--data-root", str(fixture.data),
                "--id", "mssbench/embodied/0000/0/a", "--output", str(output)])
        self.assertEqual(result, 0)
        manifest = probe.read_json(output / "manifest.json")
        self.assertEqual(manifest["status"], "ok")
        self.assertEqual(manifest["generation_parameters_sent"], [])
        self.assertIsNone(manifest["total_duration_limit"])
        self.assertTrue((output / "attempt-01/events.jsonl").is_file())
        self.assertFalse((output / ".stream-probe.lock").exists())


if __name__ == "__main__":
    unittest.main()
