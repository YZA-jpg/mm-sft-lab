"""Existing-service profiling: bounded threads and real files, mocked HTTP/GPU only."""

import contextlib
import io
import json
import sys
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "tests"))
import test_vllm_scripts as fixtures
import profile_vllm_concurrency as profile
import smoke_vllm_benchmarks as smoke
from mm_sft_lab.common import read_jsonl


class ConcurrencyTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.ScriptTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        self.root, self.data, self.server = self.fixture.root, self.fixture.data, self.fixture.server
        receipt = profile.read_json(self.server / "server.json")
        receipt["gpu"] = "4"
        profile.write_json(self.server / "server.json", receipt)
        self.receipt = receipt
        self.jobs = profile.fair_order(smoke.select_jobs(self.data, 1)[0])
        self.payloads = {row["job_id"]: smoke.make_payload(row["record"], self.data, self.fixture.manifest, "target") for row in self.jobs}

    def args(self, output):
        return ["--server-dir", str(self.server), "--data-root", str(self.data), "--output", str(output),
                "--limit", "1", "--rounds", "2", "--levels", "1", "2", "4"]

    def fake_completion(self, url, payload, timeout, offset=0, on_attempt=None):
        self.assertEqual(set(payload), {"model", "messages"})
        self.assertEqual(url, "http://127.0.0.1:18000/v1")
        if on_attempt:
            on_attempt({"number": 1, "started_at": "synthetic-time"})
        return {"status": "ok", "response": "Synthetic complete answer", "finish_reason": "stop",
            "usage": {"completion_tokens": 3}, "wall_seconds": .01, "error_kind": None,
            "http_status": 200, "request_sha256": profile.digest(payload)}

    def test_calls_overlap_but_never_exceed_the_configured_limit(self):
        lock, release = threading.Lock(), threading.Event()
        active, maximum, entered = 0, 0, 0
        def completion(*args, **kwargs):
            nonlocal active, maximum, entered
            with lock:
                active += 1
                entered += 1
                maximum = max(maximum, active)
                if entered == 2:
                    release.set()
            self.assertTrue(release.wait(3), "Requests did not overlap")
            time.sleep(.005)
            try:
                return self.fake_completion(*args, **kwargs)
            finally:
                with lock:
                    active -= 1
        folder = self.root / "bounded"
        with patch.object(profile, "completion", side_effect=completion), \
             patch.object(profile, "gpu_snapshot", return_value={"synthetic": True}), contextlib.redirect_stdout(io.StringIO()):
            result = profile.run_trial(self.jobs, self.payloads, self.receipt, folder, 2, 30, "fingerprint")
        self.assertEqual(maximum, 2)
        self.assertEqual(result["successful_requests"], 7)
        self.assertEqual(len(read_jsonl(folder / "predictions.jsonl")), 7)
        events = read_jsonl(folder / "transport-attempts.jsonl")
        self.assertEqual({r["job_id"] for r in events}, {j["job_id"] for j in self.jobs})
        self.assertEqual(len(events), 7)

    def test_two_rounds_reuse_inputs_exclude_warmup_and_reverse_order(self):
        output = self.root / "comparison"
        calls = []
        def completion(*args, **kwargs):
            calls.append(profile.digest(args[1]))
            return self.fake_completion(*args, **kwargs)
        with patch.object(profile, "completion", side_effect=completion), \
             patch.object(profile, "wait_for_server", return_value=12), \
             patch.object(profile, "gpu_snapshot", return_value={"synthetic": True}), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(profile.main(self.args(output)), 0)
        self.assertEqual(len(calls), 43)  # 7 x 3 levels x 2 rounds + one excluded warmup.
        manifest = profile.read_json(output / "manifest.json")
        self.assertEqual(manifest["expected_measured_requests"], 42)
        self.assertEqual(manifest["judge_calls"], 0)
        self.assertEqual(manifest["server_readiness_wait_seconds"], 12)
        self.assertEqual(manifest["signature"]["generation_parameters_sent"], [])
        self.assertEqual(profile.read_json(output / "warmup/summary.json")["successful_requests"], 1)
        summary = profile.read_json(output / "summary.json")
        self.assertEqual([r["concurrency"] for r in summary["trials"]], [1, 2, 4, 4, 2, 1])
        expected = {(j["job_id"], profile.digest(self.payloads[j["job_id"]])) for j in self.jobs}
        for trial in summary["trials"]:
            rows = read_jsonl(output / trial["directory"] / "predictions.jsonl")
            self.assertEqual({(r["job_id"], r["request_sha256"]) for r in rows}, expected)
            self.assertEqual(trial["output_tokens"], 21)
        for row in summary["comparison"]:
            self.assertTrue(row["performance_comparable"])
            self.assertEqual(row["successful_requests"], 14)
        self.assertTrue((output / "comparison.csv").exists())

    def test_auth_error_is_retained_and_stops_further_trials(self):
        output = self.root / "error"
        def completion(*args, **kwargs):
            row = self.fake_completion(*args, **kwargs)
            return {**row, "status": "error", "http_status": 401, "error_kind": "http_error",
                    "response": None, "usage": {}, "finish_reason": None}
        with patch.object(profile, "completion", side_effect=completion), \
             patch.object(profile, "wait_for_server", return_value=0), \
             patch.object(profile, "gpu_snapshot", return_value={"synthetic": True}), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(profile.main(self.args(output) + ["--warmup", "0"]), 2)
        self.assertEqual(profile.read_json(output / "manifest.json")["status"], "stopped_on_request_errors")
        summary = profile.read_json(output / "summary.json")
        self.assertEqual(len(summary["trials"]), 1)
        self.assertEqual(summary["trials"][0]["error_requests"], 1)
        self.assertEqual(summary["trials"][0]["missing_requests"], 6)
        self.assertTrue(all(r["median_requests_per_second"] is None for r in summary["comparison"]))

    def test_existing_outputs_are_never_overwritten(self):
        output = self.root / "occupied"
        output.mkdir()
        marker = output / "manifest.json"
        marker.write_text("preserve existing data")
        with patch.object(profile, "wait_for_server") as ready:
            with self.assertRaisesRegex(ValueError, "fresh output directory"):
                profile.main(self.args(output))
        ready.assert_not_called()
        self.assertEqual(marker.read_text(), "preserve existing data")
        self.assertFalse((output / ".concurrency.lock").exists())

    def test_missing_usage_and_incomplete_rounds_do_not_produce_misleading_rates(self):
        rows = [{"status": "ok", "response": "answer", "finish_reason": "stop", "usage": {},
                 "wall_seconds": 2, "task": "siuo/gen", "started_offset_seconds": 0, "finished_offset_seconds": 2}]
        result = profile.summarize(rows, 1, 4)
        self.assertEqual(result["successful_requests_per_second"], .25)
        self.assertIsNone(result["output_tokens_per_second"])
        rows[0]["usage"] = {"completion_tokens": "not an integer"}
        self.assertIsNone(profile.summarize(rows, 1, 4)["output_tokens_per_second"])
        values = profile.comparison([{**result, "concurrency": 1}], [1, 2], rounds=2)
        self.assertTrue(all(r["median_requests_per_second"] is None for r in values))
        rows[0]["finish_reason"] = "length"
        self.assertFalse(profile.summarize(rows, 1, 4)["performance_comparable"])

    def test_invalid_concurrency_is_rejected_before_any_requests(self):
        with contextlib.redirect_stderr(io.StringIO()):
            for levels in (["0"], ["1", "1"], ["-1"]):
                with self.assertRaises(SystemExit):
                    profile.main(["--levels", *levels])

    def test_throughput_pools_work_and_time_instead_of_averaging_round_rates(self):
        trials = []
        for level, durations in ((1, (1, 9)), (2, (2, 2))):
            for duration in durations:
                trials.append({"concurrency": level, "performance_comparable": True,
                    "successful_requests": 7, "expected_requests": 7, "wall_seconds": duration,
                    "successful_requests_per_second": 7 / duration, "output_tokens": 70,
                    "output_tokens_per_second": 70 / duration, "issue_counts": {}})
        serial, parallel = profile.comparison(trials, [1, 2], 2)
        self.assertEqual(serial["total_wall_seconds"], 10)
        self.assertAlmostEqual(serial["pooled_requests_per_second"], 1.4)
        self.assertEqual(serial["pooled_output_tokens_per_second"], 14)
        self.assertGreater(serial["median_requests_per_second"], parallel["median_requests_per_second"])
        self.assertAlmostEqual(parallel["request_throughput_relative_to_c1"], 2.5)
        incomplete = profile.comparison(trials[:1], [1], 2)[0]
        self.assertIsNone(incomplete["pooled_requests_per_second"])


if __name__ == "__main__":
    unittest.main()
