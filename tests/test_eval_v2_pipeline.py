"""Cross-benchmark counts, exclusions, resumability and protocol isolation, with no sockets."""

import contextlib
import copy
import io
import json
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation_v2.common import append, digest, read_json, read_journal, write_json
from evaluation_v2.config import validate
from evaluation_v2.data import load_dataset
from evaluation_v2.demo import FakeHTTP, make_demo
from evaluation_v2.metrics import export_report, summarize, union_seconds
from evaluation_v2.prompts import target_request
from evaluation_v2.reporting import compare
from evaluation_v2.runner import fair_order, generation_jobs, initialize, load_run, stage
from evaluation_v2.transport import Client
from evaluation_v2.status import snapshot, status


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.data, self.run = self.root / "data", self.root / "run"
        make_demo(self.data)
        self.config = validate({"execution": {"generation_concurrency": 4, "deepseek_concurrency": 3}})
        initialize(self.data, self.run, self.config, {"kind": "mock", "label": "SIMULATION"})
        self.fake = FakeHTTP()

    def tearDown(self):
        self.tmp.cleanup()

    def stages(self, fake=None):
        with contextlib.redirect_stdout(io.StringIO()):
            for phase in ("generation", "deepseek", "wildguard"):
                stage(self.run, phase, client_factory=(fake or self.fake).factory)

    def test_eight_benchmark_demo_and_resume_never_repeats_completed_jobs(self):
        self.stages()
        count = len(self.fake.calls)
        self.stages()
        self.assertEqual(count, 50)
        self.assertEqual(len(self.fake.calls), count)
        report = export_report(self.run)
        self.assertTrue(report["is_mock"])
        self.assertTrue(report["complete"])
        self.assertEqual(report["generation"]["successful"], 26)
        self.assertEqual(report["metrics"]["siuo/ASR@3"]["x"], 2)
        self.assertEqual(report["metrics"]["figstep/ASR@5"]["y"], 1)
        self.assertEqual(report["metrics"]["mmstar/Acc"]["x"], 1)
        self.assertEqual(report["metrics"]["mathvista/Acc"]["x"], 1)
        self.assertEqual(report["metrics"]["mmvet/Score"]["x"], .7)
        self.assertEqual(report["quality"]["deepseek/ALL"]["unique_attempted_jobs"], 22)
        self.assertTrue((self.run / "report/main-table.csv").exists())
        self.assertTrue((self.run / "report/quality.csv").exists())
        self.assertFalse(any("Synthetic mmstar" in json.dumps(c["body"]) for c in self.fake.calls if c["body"]["model"] != "target"))

    def test_missing_one_trial_excludes_whole_asr_group_and_retains_bounds(self):
        self.stages()
        rows = read_journal(self.run / "deepseek/results.jsonl")
        row = copy.deepcopy(next(r for r in rows if r["sample_id"] == "siuo/gen/one"))
        row.update(attempt=2, status="error", error_kind="provider_guardrail", labels=None, text=None)
        append(self.run / "deepseek/results.jsonl", row)
        report = summarize(self.run)
        metric = report["metrics"]["siuo/ASR@3"]
        self.assertEqual((metric["x"], metric["y"], metric["expected"]), (1, 1, 2))
        self.assertFalse(metric["complete"])
        self.assertEqual(metric["bounds_all_expected"], {"lower": 1.0, "upper": 1.0})
        self.assertEqual(report["quality"]["deepseek/ALL"]["provider_guardrail"]["x"], 1)
        before = len(self.fake.calls)
        self.stages()
        self.assertEqual(before, len(self.fake.calls))

    def test_format_refusal_and_null_answer_have_distinct_denominators(self):
        self.stages()
        rows = read_journal(self.run / "deepseek/results.jsonl")
        for benchmark, kind in (("siuo", "format_error"), ("holisafe", "judge_refusal")):
            row = copy.deepcopy(next(r for r in rows if r["benchmark"] == benchmark))
            row.update(attempt=2, status="error", error_kind=kind, labels=None, text="unscored")
            append(self.run / "deepseek/results.jsonl", row)
        row = copy.deepcopy(next(r for r in rows if r["benchmark"] == "mathvista"))
        row.update(attempt=2, text='{"answer":null}', labels={"answer": None})
        append(self.run / "deepseek/results.jsonl", row)
        report = summarize(self.run)
        quality = report["quality"]["deepseek/ALL"]
        self.assertEqual(quality["format_error_attempted"]["x"], 1)
        self.assertEqual(quality["judge_refusal"]["x"], 1)
        self.assertEqual(quality["format_error_attempted"]["y"], 22)
        self.assertEqual((report["metrics"]["mathvista/Acc"]["x"], report["metrics"]["mathvista/Acc"]["y"]), (0, 1))

    def test_resume_binds_model_config_and_snapshot(self):
        initialize(self.data, self.run, self.config, {"kind": "mock", "label": "SIMULATION"}, resume=True)
        different = copy.deepcopy(self.config)
        different["execution"]["generation_concurrency"] = 8
        with self.assertRaisesRegex(ValueError, "Existing run differs"):
            initialize(self.data, self.run, different, {"kind": "mock", "label": "SIMULATION"}, resume=True)
        path = self.run / "selection.jsonl"
        path.write_text(path.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "snapshot changed"):
            load_run(self.run)

    def test_stale_judge_labels_and_wrong_phase_cannot_be_reported(self):
        self.stages()
        row = copy.deepcopy(read_journal(self.run / "deepseek/results.jsonl")[0])
        row.update(attempt=2, labels={"tampered": True})
        append(self.run / "deepseek/results.jsonl", row)
        with self.assertRaisesRegex(ValueError, "labels differ"):
            summarize(self.run)

    def test_lock_prevents_final_report_while_writer_is_active(self):
        (self.run / "generation").mkdir()
        (self.run / "generation/.stage.lock").write_text("{}")
        with self.assertRaisesRegex(ValueError, "Finish generation"):
            summarize(self.run)

    def test_mss_smoke_limit_preserves_both_conditions(self):
        config = validate({"benchmarks": ["mssbench"], "execution": {"per_benchmark_limit": 1}})
        records, _ = load_dataset(self.data, config)
        self.assertEqual(len(records), 4)
        for split in ("chat", "embodied"):
            self.assertEqual({r["reference"]["condition"] for r in records if r["split"] == split}, {"safe", "unsafe"})

    def test_target_payload_never_contains_reference_labels(self):
        _, records, data = load_run(self.run)
        record = next(r for r in records if r["benchmark"] == "mmvet")
        request = target_request(record, self.data, data)
        self.assertNotIn(record["reference"]["answer"], json.dumps(request))

    def test_scheduler_fairness_and_bounded_concurrency(self):
        _, records, _ = load_run(self.run)
        first = list(fair_order(generation_jobs(records, self.config)))[:8]
        self.assertEqual(len({j["benchmark"] for j in first}), 8)
        active, maximum, guard = 0, 0, threading.Lock()
        fake = self.fake
        def opener(request, **kwargs):
            nonlocal active, maximum
            with guard:
                active += 1
                maximum = max(maximum, active)
            time.sleep(.01)
            try:
                return fake(request, **kwargs)
            finally:
                with guard:
                    active -= 1
        def factory(endpoint, execution):
            return Client({**endpoint, "api_key_env": None}, execution, opener=opener, sleeper=lambda _: None)
        with contextlib.redirect_stdout(io.StringIO()):
            stage(self.run, "generation", client_factory=factory)
        self.assertEqual(maximum, 4)

    def test_durable_attempt_budget_survives_interrupted_request(self):
        config = validate({"benchmarks": ["mmstar"], "execution": {"per_benchmark_limit": 1, "generation_concurrency": 1}})
        run = self.root / "interrupted"
        manifest = initialize(self.data, run, config, {"kind": "mock", "label": "Interrupted"})
        directory = run / "generation"
        directory.mkdir()
        key = "mmstar/test/a#g0"
        for number in range(1, 6):
            append(directory / "transport-attempts.jsonl", {"job_id": key, "number": number, "run_fingerprint": manifest["run_fingerprint"], "session_id": "interrupted"})
        with contextlib.redirect_stdout(io.StringIO()):
            stage(run, "generation", client_factory=self.fake.factory)
        self.assertEqual(len(self.fake.calls), 1)
        self.assertEqual(read_journal(directory / "transport-attempts.jsonl")[-1]["number"], 6)

    def test_compare_rejects_changed_generation_defaults(self):
        self.stages()
        report = export_report(self.run)
        left, right = self.root / "left.json", self.root / "right.json"
        write_json(left, report)
        other = copy.deepcopy(report)
        other["model_label"] = "SFT"
        write_json(right, other)
        compare([left, right], self.root / "comparison")
        other["model_identity"]["config_sha256"] = {"generation_config.json": "different"}
        write_json(right, other)
        with self.assertRaisesRegex(ValueError, "generation_config"):
            compare([left, right], self.root / "bad-comparison")

    def test_overlap_time_is_union_not_sum(self):
        self.assertEqual(union_seconds([(0, 10), (2, 3), (9, 20), (30, 35)]), 25)

    def test_live_status_ignores_unfinished_line_without_mutating_the_file(self):
        self.stages()
        path = self.run / "deepseek/results.jsonl"
        with path.open("ab") as stream:
            stream.write(b'{"partial":')
        before = path.read_bytes()
        result = status(self.run)
        self.assertEqual(result["phases"]["deepseek"]["valid"], 22)
        self.assertEqual(result["phases"]["generation"]["remaining_without_result"], 0)
        self.assertEqual(path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
