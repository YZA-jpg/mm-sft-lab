"""Fresh-run MIME, selection and serial scheduling regressions; no sockets or models."""

import base64
import contextlib
import io
import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation_v2.common import read_journal
from evaluation_v2.config import load, validate
from evaluation_v2.data import Writer, load_dataset
from evaluation_v2.demo import FakeHTTP, make_demo
from evaluation_v2.prompts import _image_url, request_preflight, target_request
from evaluation_v2.runner import generation_jobs, initialize, stage
from evaluation_v2.transport import Client
from scripts.export_eval_v2_timing import build_timing


class FixedBaselineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.data = self.root / "data"
        make_demo(self.data)
        self.addCleanup(_image_url.cache_clear)

    def config(self, **execution):
        return validate({"benchmarks": ["siuo", "mssbench"], "generation": {"max_tokens": 65536},
                         "execution": {"benchmark_schedule": "serial", **execution}})

    def test_webp_is_independent_of_system_mime_and_preserves_original_bytes(self):
        out = io.BytesIO()
        Image.new("RGB", (32, 32), "blue").save(out, format="WEBP")
        payload = out.getvalue()
        writer = Writer(self.root / "webp")
        image = writer.image(payload, self.root)
        row = {"id": "siuo/gen/a", "benchmark": "siuo", "split": "gen", "query": "Synthetic query",
               "images": [image], "reference": {}}
        manifest = writer.finish([row], is_mock=True)
        with patch("mimetypes.guess_type", return_value=(None, None)):
            request = target_request(row, writer.root, manifest)
            result = request_preflight([row], writer.root, manifest)
        url = request["messages"][0]["content"][0]["image_url"]["url"]
        self.assertTrue(url.startswith("data:image/webp;base64,"))
        self.assertEqual(base64.b64decode(url.split(",", 1)[1]), payload)
        self.assertEqual((writer.root / image).read_bytes(), payload)
        self.assertEqual(result["unique_image_mime_counts"], {"image/webp": 1})
        self.assertEqual(result["samples_checked"], 1)

    def test_unknown_image_extension_fails_preflight_before_run_creation(self):
        rows, data = load_dataset(self.data, self.config())
        row = dict(rows[0], images=["assets/unsupported.bmp"])
        with self.assertRaisesRegex(ValueError, "Unsupported image extension"):
            request_preflight([row], self.data, data)
        with patch("evaluation_v2.runner.request_preflight", side_effect=ValueError("bad image")):
            with self.assertRaisesRegex(ValueError, "bad image"):
                initialize(self.data, self.root / "bad-run", self.config(), {"kind": "mock"})
        self.assertFalse((self.root / "bad-run/manifest.json").exists())

    def test_selected_sample_expands_the_mss_pair_and_preserves_trials(self):
        config = self.config(sample_ids=["siuo/gen/one", "mssbench/chat/unsafe"])
        rows, _ = load_dataset(self.data, config)
        self.assertEqual({r["id"] for r in rows}, {"siuo/gen/one", "mssbench/chat/safe", "mssbench/chat/unsafe"})
        jobs = generation_jobs(rows, config)
        self.assertEqual(len(jobs), 5)
        self.assertEqual([j["trial"] for j in jobs if j["benchmark"] == "siuo"], [0, 1, 2])
        with self.assertRaisesRegex(ValueError, "Unknown or unselected sample IDs"):
            load_dataset(self.data, self.config(sample_ids=["siuo/gen/missing"]))

    def test_selection_and_schedule_reject_ambiguous_settings(self):
        for execution in ({"benchmark_schedule": "parallel"}, {"sample_ids": []}, {"sample_ids": "a"},
                          {"sample_ids": ["a", "a"]}, {"sample_ids": [3]},
                          {"sample_ids": ["a"], "per_benchmark_limit": 1}):
            with self.subTest(execution=execution), self.assertRaises(ValueError):
                self.config(**execution)

    def test_serial_barrier_keeps_four_requests_inside_one_benchmark(self):
        config, fake, run = self.config(), FakeHTTP(), self.root / "run"
        initialize(self.data, run, config, {"kind": "mock"})
        active, maximum, seen = set(), 0, []
        guard = threading.Lock()
        first_four = threading.Barrier(4)
        def opener(request, **kwargs):
            nonlocal maximum
            body = json.loads(request.data)
            text = body["messages"][0]["content"][-1]["text"]
            benchmark = "siuo" if "siuo" in text else "mssbench"
            token = object()
            with guard:
                self.assertTrue(all(b == benchmark for _, b in active))
                active.add((token, benchmark))
                maximum = max(maximum, len(active))
                seen.append(benchmark)
                number = len(seen)
            try:
                if number <= 4:
                    first_four.wait(timeout=3)
                time.sleep(.005)
                return fake(request, **kwargs)
            finally:
                with guard:
                    active.remove((token, benchmark))
        def factory(endpoint, execution):
            return Client({**endpoint, "api_key_env": None}, execution, opener=opener)
        with contextlib.redirect_stdout(io.StringIO()):
            result = stage(run, "generation", client_factory=factory)
            stage(run, "generation", client_factory=factory)
        self.assertTrue(result["scoring_complete"])
        self.assertEqual(maximum, 4)
        self.assertEqual(seen, ["siuo"] * 6 + ["mssbench"] * 4)
        rows = read_journal(run / "generation/results.jsonl")
        end = max(r["job_finished_epoch"] for r in rows if r["benchmark"] == "siuo")
        start = min(r["job_started_epoch"] for r in rows if r["benchmark"] == "mssbench")
        self.assertLessEqual(end, start)
        events = read_journal(run / "generation/benchmark-sessions.jsonl")
        completed = [e for e in events if e["event"] == "completed"]
        self.assertEqual([e["benchmark"] for e in completed], ["siuo", "mssbench"])
        self.assertTrue(all(e["processing_complete"] for e in completed))
        timing = build_timing(run)
        self.assertEqual(timing["benchmark_schedule"], "serial")
        measured = [r for r in timing["rows"] if r["phase"] == "generation" and r["split"] == "ALL" and r["benchmark"] != "ALL"]
        self.assertTrue(all(r["isolated_benchmark_wall_seconds"] > 0 for r in measured))

    def test_serial_fatal_error_does_not_start_next_benchmark(self):
        run, config = self.root / "fatal", self.config()
        initialize(self.data, run, config, {"kind": "mock"})
        with patch("evaluation_v2.runner.target_request", side_effect=ValueError("synthetic preparation failure")):
            with contextlib.redirect_stdout(io.StringIO()):
                result = stage(run, "generation", client_factory=FakeHTTP().factory)
        self.assertTrue(result["fatal"])
        self.assertTrue(all(r["benchmark"] == "siuo" for r in read_journal(run / "generation/results.jsonl")))
        events = read_journal(run / "generation/benchmark-sessions.jsonl")
        self.assertEqual(events[-1]["event"], "failed")

    def test_checked_in_fixed_configs_keep_budget_repeats_and_serial_order(self):
        repo = Path(__file__).resolve().parents[1]
        smoke = load(repo / "configs/eval_v2.siuo_mss_fixed_smoke.json")
        full = load(repo / "configs/eval_v2.siuo_mss_fixed_full.json")
        self.assertEqual(len(smoke["execution"]["sample_ids"]), 7)
        self.assertIsNone(full["execution"]["sample_ids"])
        for config in (smoke, full):
            self.assertEqual(config["benchmarks"], ["siuo", "mssbench"])
            self.assertEqual(config["repeats"]["siuo"], 3)
            self.assertEqual(config["repeats"]["mssbench"], 1)
            self.assertEqual(config["generation"], {"max_tokens": 65536})
            self.assertEqual(config["execution"]["benchmark_schedule"], "serial")
            self.assertEqual(config["execution"]["generation_concurrency"], 4)
            self.assertTrue(config["execution"]["generation_stream"])


if __name__ == "__main__":
    unittest.main()
