"""Missing-WebP-MIME recovery without changing frozen evaluator files or using sockets."""

import base64
import contextlib
import copy
import io
import json
import mimetypes
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation_v2.common import append, code_identity, file_hash, latest, read_json, read_journal
from evaluation_v2.config import validate
from evaluation_v2.data import Writer
from evaluation_v2.demo import FakeHTTP
from evaluation_v2.prompts import _image_url, target_request
from evaluation_v2.runner import initialize, stage
from evaluation_v2.transport import Client
from scripts import resume_eval_v2_webp as recovery


class WebPRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.data, self.run = self.root / "data", self.root / "run"
        writer = Writer(self.data)
        self.records = []
        for index, fmt in enumerate(("PNG", "WEBP")):
            stream = io.BytesIO()
            Image.new("RGB", (32, 32), "blue").save(stream, format=fmt)
            image = writer.image(stream.getvalue(), self.root)
            self.records.append({"id": f"mssbench/chat/0000/0/{'ab'[index]}", "benchmark": "mssbench",
                "split": "chat", "query": "Synthetic test question", "images": [image],
                "reference": {"condition": ("safe", "unsafe")[index], "pair_id": "pair"}})
        self.data_manifest = writer.finish(self.records, is_mock=True)
        self.config = validate({"benchmarks": ["mssbench"], "generation": {"max_tokens": 65536},
                                "execution": {"generation_concurrency": 1}})
        self.identity = {"kind": "local_checkpoint", "label": "fixture", "path": str(self.root / "model"),
            "serving_runtime": {"python": "/venv-symlink/bin/python", "gpu": "1",
                "memory_utilization": .65, "target_kv_cache_gib": 40, "wildguard_kv_cache_gib": None}}
        initialize(self.data, self.run, self.config, self.identity)
        self.fake = FakeHTTP()
        mimetypes.init()
        self.mime_patch = patch.dict(mimetypes.types_map)
        self.mime_patch.start()
        self.addCleanup(self.mime_patch.stop)
        mimetypes.types_map.pop(".webp", None)
        _image_url.cache_clear()
        self.addCleanup(_image_url.cache_clear)

    def fail_once(self):
        # Inject the historical OS-dependent lookup: the current core is already fixed.
        with patch("evaluation_v2.prompts.image_media_type", side_effect=lambda path: mimetypes.guess_type(path)[0]), \
             contextlib.redirect_stdout(io.StringIO()):
            result = stage(self.run, "generation", client_factory=self.fake.factory)
        self.assertTrue(result["fatal"])
        self.assertEqual(result["successful_jobs"], 1)
        self.assertEqual(len(self.fake.calls), 1)
        return latest(read_journal(self.run / "generation/results.jsonl"))

    def test_missing_webp_failure_reproduces_before_http(self):
        rows = self.fail_once()
        error = rows[self.records[1]["id"] + "#g0"]
        self.assertEqual(error["error_kind"], "local_preparation_error")
        self.assertIn("NoneType", error["error"])
        self.assertEqual(error["attempts"], [])

    def test_recovery_preserves_successful_requests_history_and_frozen_code(self):
        old = self.fail_once()
        manifest_bytes = (self.run / "manifest.json").read_bytes()
        journal_before = (self.run / "generation/results.jsonl").read_bytes()
        code_before = code_identity()
        receipt_path, receipt = recovery.prepare_recovery(self.run)
        self.assertEqual(receipt["successful_preserved"], 1)
        self.assertEqual(receipt["pending_jobs_including_repair"], 1)
        self.assertEqual(receipt["verified_successful_request_hashes"], 1)
        self.assertEqual(receipt["webp_mime_before"], None)
        self.assertEqual(receipt["webp_mime_after"], "image/webp")
        self.assertEqual(receipt["code_sha256"], code_before["sha256"])
        self.assertEqual((self.run / "manifest.json").read_bytes(), manifest_bytes)
        self.assertEqual((self.run / "generation/results.jsonl").read_bytes(), journal_before)
        self.assertEqual(code_identity(), code_before)
        self.assertTrue(receipt_path.exists())
        self.assertEqual(read_json(receipt_path.parent / "resolved-config.json"), self.config)
        with contextlib.redirect_stdout(io.StringIO()):
            result = stage(self.run, "generation", client_factory=self.fake.factory, retry_recoverable=True)
        self.assertEqual(result["successful_jobs"], 2)
        self.assertEqual(len(self.fake.calls), 2)
        history = read_journal(self.run / "generation/results.jsonl")
        self.assertEqual(len(history), 3)
        self.assertEqual(history[0], old[self.records[0]["id"] + "#g0"])
        self.assertEqual(history[1]["status"], "error")
        self.assertEqual(history[2]["attempt"], 2)
        self.assertEqual(history[2]["attempts"][0]["number"], 1)
        self.assertEqual((self.run / "manifest.json").read_bytes(), manifest_bytes)

    def test_webp_bytes_and_budget_are_not_changed(self):
        self.fail_once()
        recovery.prepare_recovery(self.run)
        request = target_request(self.records[1], self.data, self.data_manifest)
        url = request["messages"][0]["content"][0]["image_url"]["url"]
        self.assertTrue(url.startswith("data:image/webp;base64,"))
        self.assertEqual(base64.b64decode(url.split(",", 1)[1]),
                         (self.data / self.records[1]["images"][0]).read_bytes())
        args = recovery.resume_arguments(self.run, self.run / "resolved.json")
        self.assertIn("--resume", args)
        self.assertIn("--generate-only", args)
        self.assertEqual(args[args.index("--vllm-python") + 1], "/venv-symlink/bin/python")
        self.assertEqual(args[args.index("--gpu") + 1], "1")
        self.assertEqual(self.config["generation"], {"max_tokens": 65536})

    def test_unrelated_errors_are_refused_without_rewriting_journal(self):
        rows = self.fail_once()
        row = copy.deepcopy(rows[self.records[1]["id"] + "#g0"])
        row.update(attempt=2, error="StreamJournalError: [Errno 28] No space left on device")
        append(self.run / "generation/results.jsonl", row)
        before = file_hash(self.run / "generation/results.jsonl")
        with self.assertRaisesRegex(ValueError, "Unrelated"):
            recovery.prepare_recovery(self.run)
        self.assertEqual(file_hash(self.run / "generation/results.jsonl"), before)
        self.assertIsNone(mimetypes.guess_type("image.webp")[0])

    def test_prior_request_hash_mismatch_refuses_resume(self):
        rows = self.fail_once()
        row = copy.deepcopy(rows[self.records[0]["id"] + "#g0"])
        row.update(attempt=2, request_sha256="0" * 64)
        append(self.run / "generation/results.jsonl", row)
        with self.assertRaisesRegex(ValueError, "request hash"):
            recovery.prepare_recovery(self.run)

    def test_live_lock_refuses_preparation(self):
        self.fail_once()
        (self.run / ".pipeline.lock").write_text("{}")
        with self.assertRaisesRegex(ValueError, "locked"):
            recovery.prepare_recovery(self.run)

    def test_check_only_never_delegates_model_execution(self):
        self.fail_once()
        with patch.object(recovery.cli, "main", side_effect=AssertionError("No model execution")), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(recovery.main(["--run-dir", str(self.run), "--check-only"]), 0)
        self.assertEqual(len(self.fake.calls), 1)

    def test_resume_delegates_original_cli_with_frozen_arguments(self):
        self.fail_once()
        with patch.object(recovery.cli, "main", return_value=0) as original, \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(recovery.main(["--run-dir", str(self.run)]), 0)
        argv = original.call_args.args[0]
        self.assertEqual(argv[0], "run")
        config = read_json(Path(argv[argv.index("--config") + 1]))
        self.assertEqual(config, self.config)
        self.assertIn("--resume", argv)
        self.assertIn("--generate-only", argv)

    def test_streaming_request_identity_includes_budget_and_usage_flags(self):
        config = copy.deepcopy(self.config)
        config["execution"]["generation_stream"] = True
        run = self.root / "stream-run"
        initialize(self.data, run, config, self.identity)
        _image_url.cache_clear()
        calls = []

        def opener(request, **kwargs):
            body = json.loads(request.data)
            calls.append(body)
            chunk = {"choices": [{"delta": {"content": "A"}, "finish_reason": "stop"}],
                     "usage": {"completion_tokens": 5}}
            return io.BytesIO(("data: " + json.dumps(chunk) + "\n\ndata: [DONE]\n\n").encode())

        def factory(endpoint, execution):
            return Client({**endpoint, "api_key_env": None}, execution, opener=opener)

        with patch("evaluation_v2.prompts.image_media_type", side_effect=lambda path: mimetypes.guess_type(path)[0]), \
             contextlib.redirect_stdout(io.StringIO()):
            result = stage(run, "generation", client_factory=factory)
        self.assertTrue(result["fatal"])
        receipt_path, receipt = recovery.prepare_recovery(run)
        self.assertEqual(receipt["verified_successful_request_hashes"], 1)
        self.assertEqual(calls[0]["max_tokens"], 65536)
        self.assertEqual(calls[0]["stream_options"], {"include_usage": True, "continuous_usage_stats": True})
        with contextlib.redirect_stdout(io.StringIO()):
            resumed = stage(run, "generation", client_factory=factory, retry_recoverable=True)
        self.assertEqual(resumed["successful_jobs"], 2)
        self.assertEqual(len(calls), 2)


if __name__ == "__main__":
    unittest.main()

