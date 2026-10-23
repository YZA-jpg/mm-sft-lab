"""Timing semantics for concurrent/interrupted runs; synthetic journals only, no sockets."""

import copy
import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation_v2.common import append, file_hash, read_journal
from evaluation_v2.config import validate
from evaluation_v2.demo import make_demo
from evaluation_v2.runner import generation_jobs, initialize
from scripts.export_eval_v2_timing import build_timing, export_timing


class DatasetTimingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.data, self.run = self.root / "data", self.root / "run"
        make_demo(self.data)
        self.config = validate({"benchmarks": ["siuo", "mssbench"]})
        manifest = initialize(self.data, self.run, self.config, {"kind": "mock", "label": "TIMING FIXTURE"})
        self.fingerprint = manifest["run_fingerprint"]
        records = read_journal(self.run / "selection.jsonl")
        self.jobs = {r["job_id"]: r for r in generation_jobs(records, self.config)}
        self.path = self.run / "generation/results.jsonl"
        self.path.parent.mkdir()
        self.write_row("siuo/gen/one#g0", 0, 5, "A", tokens=5)
        self.write_row("mssbench/chat/safe#g0", 3, 10, "A", tokens=6,
                       attempts=[{"status": "error", "elapsed_seconds": 2},
                                 {"status": "ok", "elapsed_seconds": 4}], backoff=1)
        self.write_row("mssbench/embodied/unsafe#g0", 5, 9, "A", ok=False)
        self.write_row("siuo/gen/one#g1", 100, 104, "B", tokens=4)
        for index in (1, 2):
            append(self.run / "generation/transport-attempts.jsonl",
                   {"job_id": "mssbench/chat/safe#g0", "number": index, "run_fingerprint": self.fingerprint})
        append(self.run / "pipeline-sessions.jsonl", {"event": "started", "session_id": "failed-pipeline"})
        for session, duration in (("A", 10), ("B", 4)):
            append(self.run / "generation/sessions.jsonl", {"event": "started", "session_id": session})
            append(self.run / "generation/sessions.jsonl",
                   {"event": "completed", "session_id": session, "wall_seconds": duration})

    def write_row(self, job_id, start, end, session, *, tokens=0, ok=True, attempt=1, attempts=None, backoff=0):
        job = self.jobs[job_id]
        row = {k: v for k, v in job.items() if k != "record"}
        row.update(run_fingerprint=self.fingerprint, session_id=session, attempt=attempt,
                   status="ok" if ok else "error", error_kind=None if ok else "local_preparation_error",
                   job_started_epoch=start, job_finished_epoch=end, job_wall_seconds=end - start,
                   usage={"completion_tokens": tokens} if ok else {}, finish_reason="stop" if ok else None,
                   attempts=attempts or [], backoff_seconds=backoff)
        append(self.path, row)

    def row(self, report, benchmark, split="ALL", phase="generation"):
        return next(r for r in report["rows"]
                    if (r["phase"], r["benchmark"], r["split"]) == (phase, benchmark, split))

    def test_overlap_and_restart_gaps_are_not_summed_as_elapsed(self):
        report = build_timing(self.run)
        overall = self.row(report, "ALL")
        self.assertEqual(overall["observed_span_seconds_excluding_restart_gaps"], 14)
        self.assertEqual(overall["active_interval_union_seconds"], 14)
        self.assertEqual(overall["sum_job_wall_seconds"], 20)
        siuo = self.row(report, "siuo")
        mss = self.row(report, "mssbench")
        self.assertEqual(siuo["observed_span_seconds_excluding_restart_gaps"], 9)
        self.assertEqual(mss["observed_span_seconds_excluding_restart_gaps"], 7)
        self.assertGreater(9 + 7, overall["observed_span_seconds_excluding_restart_gaps"])
        self.assertEqual(overall["expected_jobs"], 10)
        self.assertEqual((overall["handled_unique_jobs"], overall["valid_unique_jobs"], overall["error_unique_jobs"]),
                         (4, 3, 1))
        self.assertEqual(overall["unattempted_jobs"], 6)
        self.assertFalse(overall["processing_complete"])

    def test_retry_and_failure_costs_and_modules_are_retained(self):
        report = build_timing(self.run)
        chat = self.row(report, "mssbench", "chat")
        embodied = self.row(report, "mssbench", "embodied")
        self.assertEqual(chat["sum_http_attempt_seconds"], 6)
        self.assertEqual(chat["sum_backoff_seconds"], 1)
        self.assertEqual(chat["transport_attempts_started"], 2)
        self.assertEqual(chat["failed_http_attempts_recorded"], 1)
        self.assertEqual(embodied["sum_job_wall_seconds"], 4)
        self.assertEqual(embodied["error_unique_jobs"], 1)
        self.assertEqual(report["pipeline"]["completed_session_wall_seconds"], 0)
        self.assertEqual(report["pipeline"]["sessions_without_completed_event"], ["failed-pipeline"])
        self.assertEqual(report["phase_sessions"]["generation"]["completed_session_wall_seconds"], 14)

    def test_recovery_keeps_old_execution_cost_but_counts_latest_job_once(self):
        self.write_row("mssbench/embodied/unsafe#g0", 105, 109, "B", tokens=10, attempt=2)
        report = build_timing(self.run)
        total = self.row(report, "ALL")
        self.assertEqual(total["recorded_job_executions"], 5)
        self.assertEqual(total["handled_unique_jobs"], 4)
        self.assertEqual(total["valid_unique_jobs"], 4)
        self.assertEqual(total["error_unique_jobs"], 0)
        self.assertEqual(total["sum_job_wall_seconds"], 24)
        self.assertEqual(total["active_interval_union_seconds"], 18)
        self.assertEqual(total["observed_span_seconds_excluding_restart_gaps"], 19)
        self.assertEqual(total["latest_valid_completion_tokens"], 25)

    def test_live_partial_line_is_ignored_without_repair_or_mutation(self):
        with self.path.open("ab") as stream:
            stream.write(b'{"unfinished":')
        before = file_hash(self.path)
        (self.run / "generation/.stage.lock").write_text("{}")
        output, report = export_timing(self.run, self.root / "output")
        self.assertEqual(file_hash(self.path), before)
        self.assertEqual(self.row(report, "ALL")["handled_unique_jobs"], 4)
        self.assertTrue(report["phase_sessions"]["generation"]["stage_lock_present"])
        self.assertEqual(report["scope"], "read_only_timing_snapshot_not_final_scores")
        self.assertTrue((output / "dataset-timing.csv").exists())

    def test_csv_no_overwrite_and_source_manifest_unchanged(self):
        before = file_hash(self.run / "manifest.json")
        out, report = export_timing(self.run, self.root / "csv")
        with (out / "dataset-timing.csv").open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        self.assertTrue(any(r["benchmark"] == "mssbench" and r["split"] == "chat" for r in rows))
        self.assertTrue(any(r["phase"] == "deepseek" and r["valid_unique_jobs"] == "0" for r in rows))
        self.assertFalse(any(r["phase"] == "wildguard" for r in rows))
        self.assertEqual(file_hash(self.run / "manifest.json"), before)
        saved = file_hash(out / "dataset-timing.json")
        with self.assertRaises(FileExistsError):
            export_timing(self.run, out)
        self.assertEqual(file_hash(out / "dataset-timing.json"), saved)

    def test_mixed_run_and_changed_selection_are_rejected(self):
        other = copy.deepcopy(next(iter(self.jobs.values())))
        other.pop("record")
        other.update(run_fingerprint="wrong")
        append(self.path, other)
        with self.assertRaisesRegex(ValueError, "Mixed run"):
            build_timing(self.run)
        (self.run / "selection.jsonl").write_text("changed")
        with self.assertRaisesRegex(ValueError, "snapshot changed"):
            build_timing(self.run)


if __name__ == "__main__":
    unittest.main()

