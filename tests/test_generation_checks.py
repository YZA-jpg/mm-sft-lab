"""Offline acceptance tests for smoke diagnostics and batch comparisons."""

import contextlib
import io
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scripts.make_demo import make_demo
from mm_sft_lab.backends import MockBackend
from mm_sft_lab.cli import main
from mm_sft_lab.common import digest, read_jsonl, stage_lock, write_jsonl
from mm_sft_lab.config import load_config
from mm_sft_lab.diagnostics import compare_batches, validate_run
from mm_sft_lab.runner import generate

REPO = Path(__file__).resolve().parents[1]


class GenerationCheckTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.data, self.before, self.after = (self.root / name for name in ("data", "before", "after"))
        make_demo(self.data)
        self.config = load_config(REPO / "configs/mock.json")
        quiet = contextlib.redirect_stdout(io.StringIO())
        quiet.__enter__()
        self.addCleanup(quiet.__exit__, None, None, None)

    def generated(self, *, batched=False, **kwargs):
        return generate({**self.config, "batch_size": 4 if batched else 1}, self.data,
                        self.after if batched else self.before, **kwargs)

    def edit_predictions(self, edit, root=None):
        path = (root or self.before) / "predictions.jsonl"
        rows = read_jsonl(path)
        edit(rows)
        for row in rows:
            if row["status"] == "ok":
                row["response_sha256"] = digest(row["response"])
        write_jsonl(path, rows)

    def compare(self):
        return compare_batches(self.before, self.after, self.root / "comparison")

    def test_mock_validation_is_complete_but_cannot_pass_real_model_check(self):
        self.generated()
        result = validate_run(self.before)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["complete_mssbench_pairs"], 2)
        self.assertFalse(result["real_local_generation_passed"])
        self.assertTrue(result["is_mock"])
        self.assertEqual(len(read_jsonl(self.before / "checks/review.jsonl")), 6)
        self.assertTrue((self.before / "checks/review.csv").is_file())
        self.assertFalse((self.before / "judge").exists())

    def test_batched_timing_counts_each_batch_once(self):
        self.generated(batched=True)
        rows = read_jsonl(self.after / "predictions.jsonl")
        for row in rows:
            row["batch_wall_ms"] = 1000
            row["output_tokens"] = 10
        write_jsonl(self.after / "predictions.jsonl", rows)
        result = validate_run(self.after)["performance"]
        self.assertEqual(result["actual_batch_counts"], {"4": 1, "2": 1})
        self.assertEqual(result["successful_batch_wall_seconds"], 2)
        self.assertEqual(result["requests_per_second"], 3)
        self.assertEqual(result["output_tokens_per_second"], 30)

    def test_incomplete_pair_stops_validation_and_cli(self):
        self.generated()
        self.edit_predictions(lambda rows: rows.pop())
        result = validate_run(self.before)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["missing_requests"], 1)
        self.assertEqual(result["complete_mssbench_pairs"], 1)
        self.assertEqual(main(["validate-run", "--run-dir", str(self.before)]), 2)

    def test_empty_response_is_failure(self):
        self.generated()
        self.edit_predictions(lambda rows: rows[0].update(response=" \n"))
        result = validate_run(self.before)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["issue_counts"]["empty_response"], 1)

    def test_truncation_and_invalid_mcqa_require_review(self):
        self.generated()
        def edit(rows):
            rows[0]["finish_reason"] = "length"
            next(row for row in rows if "/mcqa/" in row["id"])["response"] = "A or B"
        self.edit_predictions(edit)
        result = validate_run(self.before)
        self.assertEqual(result["status"], "needs_review")
        self.assertEqual(result["issue_counts"], {"invalid_mcqa": 1, "truncated": 1})

    def test_response_hash_corruption_is_rejected(self):
        self.generated()
        rows = read_jsonl(self.before / "predictions.jsonl")
        rows[0]["response"] = "modified outside journal contract"
        write_jsonl(self.before / "predictions.jsonl", rows)
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            validate_run(self.before)

    def test_wrong_model_or_prompt_metadata_is_failure(self):
        self.generated()
        self.edit_predictions(lambda rows: rows[0].update(model_fingerprint="different checkpoint"))
        self.assertEqual(validate_run(self.before)["issue_counts"]["prediction_identity_mismatch"], 1)

    def test_active_writer_is_rejected(self):
        self.generated()
        with stage_lock(self.before, "generate"):
            with self.assertRaisesRegex(ValueError, "locked"):
                validate_run(self.before)

    def test_same_greedy_outputs_pass_batch_comparison(self):
        self.generated()
        self.generated(batched=True)
        result = self.compare()
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["changed_requests"], 0)
        self.assertTrue(result["configured_batch_observed"])
        self.assertFalse(result["real_local_batch_check_passed"])

    def test_text_difference_is_exported_for_review(self):
        self.generated()
        self.generated(batched=True)
        self.edit_predictions(lambda rows: rows[0].update(response="Different answer."), self.after)
        result = self.compare()
        self.assertEqual(result["status"], "needs_review")
        self.assertEqual(result["change_counts"]["normalized_text_changed"], 1)
        review = read_jsonl(self.root / "comparison/review.jsonl")
        self.assertEqual(review[0]["after_response"], "Different answer.")
        self.assertIn("query", review[0])

    def test_whitespace_difference_is_distinguished_from_content_difference(self):
        self.generated()
        self.generated(batched=True)
        self.edit_predictions(lambda rows: rows[0].update(response=rows[0]["response"] + "\n"), self.after)
        result = self.compare()
        self.assertEqual(result["change_counts"], {"text_changed": 1})

    def test_config_changes_other_than_batch_size_are_rejected(self):
        self.generated()
        generate({**self.config, "batch_size": 4, "max_new_tokens": 17}, self.data, self.after)
        with self.assertRaisesRegex(ValueError, "same config"):
            self.compare()

    def test_incomplete_comparison_cannot_pass(self):
        self.generated()
        self.generated(batched=True)
        self.edit_predictions(lambda rows: rows.pop(), self.after)
        result = self.compare()
        self.assertEqual(result["status"], "needs_review")
        self.assertEqual(result["change_counts"]["incomplete_response"], 1)

    def test_actual_batch_size_cannot_be_claimed_by_a_config_alone(self):
        self.generated()
        self.generated(batched=True)
        def edit(rows):
            for index, row in enumerate(rows):
                row["batch_id"] = str(index)
                row["actual_batch_size"] = 1
        self.edit_predictions(edit, self.after)
        result = self.compare()
        self.assertFalse(result["configured_batch_observed"])
        self.assertEqual(result["status"], "needs_review")

    def test_partial_batch_journal_does_not_invent_performance(self):
        self.generated(batched=True)
        def edit(rows):
            rows[0]["batch_id"] = "interrupted-group"
        self.edit_predictions(edit, self.after)
        result = validate_run(self.after)
        self.assertEqual(result["status"], "needs_review")
        self.assertEqual(result["issue_counts"]["partial_batch_telemetry"], 2)
        self.assertEqual(result["performance"]["actual_batch_counts"], {"2": 1})

    def test_visual_grid_difference_is_visible_even_when_text_matches(self):
        backend = self.fake_local_backend()
        self.config["backend"] = "qwen3_vl_hf"
        with patch("mm_sft_lab.runner.inference_identity", return_value={"fingerprint": "fixture-model"}):
            self.generated(backend_factory=lambda config: backend)
            self.generated(batched=True, backend_factory=lambda config: backend)
        self.edit_predictions(lambda rows: rows[0].update(image_grid_thw=[[1, 4, 2]]), self.after)
        result = self.compare()
        self.assertEqual(result["status"], "needs_review")
        self.assertEqual(result["change_counts"], {"image_grid_thw_changed": 1})

    def fake_local_backend(self):
        backend = MockBackend(self.config)
        backend.metadata["fingerprint"] = "fixture-model"
        normal = backend.generate
        def with_visual_inputs(records, root, config, seeds):
            return [{**row, "input_tokens": 4, "padded_input_tokens": 7, "pad_tokens": 3, "output_tokens": 3,
                     "pixel_values_present": True, "image_count": 1, "image_grid_thw": [[1, 2, 2]],
                     "peak_cuda_allocated_bytes": 1024, "peak_cuda_reserved_bytes": 2048}
                    for row in normal(records, root, config, seeds)]
        backend.generate = with_visual_inputs
        return backend

    def test_local_telemetry_checks_detect_missing_vision_and_bad_padding(self):
        backend = self.fake_local_backend()
        self.config["backend"] = "qwen3_vl_hf"
        with patch("mm_sft_lab.runner.inference_identity", return_value={"fingerprint": "fixture-model"}):
            self.generated(backend_factory=lambda config: backend)
        result = validate_run(self.before)
        self.assertEqual(result["status"], "passed")
        self.edit_predictions(lambda rows: rows[0].update(pixel_values_present=False, pad_tokens=0))
        result = validate_run(self.before)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["issue_counts"]["invalid_visual_inputs"], 1)
        self.assertEqual(result["issue_counts"]["invalid_padding"], 1)

    def test_oom_splits_are_visible_and_do_not_pass_batch_comparison(self):
        class OutOfMemoryError(RuntimeError):
            pass
        backend = self.fake_local_backend()
        normal = backend.generate
        self.config["backend"] = "qwen3_vl_hf"
        def fail_batch(records, root, config, seeds):
            if len(records) > 1:
                raise OutOfMemoryError("fixture pressure")
            return normal(records, root, config, seeds)
        backend.torch = types.SimpleNamespace(cuda=types.SimpleNamespace(empty_cache=Mock()))
        with patch("mm_sft_lab.runner.inference_identity", return_value={"fingerprint": "fixture-model"}):
            self.generated(backend_factory=lambda config: backend)
            backend.generate = fail_batch
            manifest = self.generated(batched=True, backend_factory=lambda config: backend)
        result = self.compare()
        self.assertGreater(manifest["oom_batch_splits"], 0)
        self.assertEqual(result["after_validation"]["performance"]["actual_batch_counts"], {"1": 6})
        self.assertFalse(result["configured_batch_observed"])
        self.assertEqual(result["status"], "needs_review")

    def test_shipped_phase_configs_differ_only_in_batch_size_or_limit(self):
        smoke = load_config(REPO / "configs/baseline_smoke.json")
        batch = load_config(REPO / "configs/baseline_smoke_batch4.json")
        small = load_config(REPO / "configs/baseline_small.json")
        self.assertEqual({**smoke, "batch_size": 4}, batch)
        self.assertEqual({**batch, "per_task_limit": 8}, small)


if __name__ == "__main__":
    unittest.main()
