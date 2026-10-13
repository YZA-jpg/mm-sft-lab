"""Offline checks for benchmark identity, pairing, leakage, and provenance."""

from __future__ import annotations

import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from PIL import Image

from scripts.data_utils import SOURCES, relative_media_path, sha256_file, write_json_atomic
from scripts.download_datasets import download, get_source_lock
from scripts.prepare_datasets import build_records, prepare


class DataPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.annotations = self.root / "raw/siuo/annotations"
        self.mss_path = self.root / "raw/mssbench/combined.json"
        self.gen = [{
            "question_id": 1001, "image": "S-01.png", "question": "Describe this scene.",
            "category": "fixture-category", "safety_warning": "EVAL_WARNING",
            "reference_answer": "EVAL_REFERENCE",
        }]
        self.mcqa = [{
            "question_id": 1001, "image": "S-01.png", "question": "Describe this scene.",
            "category": "fixture-category", "options": {"A": "First choice", "B": "Second choice"},
            "correct_option": "B",
        }]
        self.mss = {
            "chat": [{
                "safe_image_path": "safe.png", "unsafe_image_path": "unsafe.png",
                "queries": ["Describe the object.", "What is on the table?"],
                "Type": ["fixture-category"], "intent": "EVAL_INTENT",
                "unsafe_image": "EVAL_CAPTION",
            }],
            "embodied": [{
                "safe": "safe.png", "unsafe": "unsafe.png", "category": "fixture-category",
                "safe_instructions": ["Move the blue item.", "Inspect the open box."],
                "unsafe_instructions": ["Move the red item.", "Inspect the closed box."],
                "safe_observations": "EVAL_OBSERVATION",
            }],
        }
        self.save_annotations()
        for relative, color in (
            ("raw/siuo/images/nested/S-01.png", "white"),
            ("raw/mssbench/chat/safe.png", "blue"),
            ("raw/mssbench/chat/unsafe.png", "red"),
            ("raw/mssbench/embodied/safe.png", "green"),
            ("raw/mssbench/embodied/unsafe.png", "yellow"),
        ):
            self.image(relative, color)

    def image(self, relative, color):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 6), color).save(path)

    def save_annotations(self):
        write_json_atomic(self.annotations / "siuo_gen.json", self.gen)
        write_json_atomic(self.annotations / "siuo_mcqa.json", self.mcqa)
        write_json_atomic(self.mss_path, self.mss)

    def source_lock(self):
        return {
            "schema_version": 1,
            "sources": {name: {**source, "commit": str(i) * 40, "requested_revision": "main"}
                        for i, (name, source) in enumerate(SOURCES.items(), start=1)},
        }

    def save_provenance(self):
        lock = self.source_lock()
        write_json_atomic(self.root / "sources.lock.json", lock)
        paths = [self.annotations / "siuo_gen.json", self.annotations / "siuo_mcqa.json", self.mss_path]
        write_json_atomic(self.root / "download_manifest.json", {
            "schema_version": 1, "sources": lock["sources"],
            "sources_lock_sha256": sha256_file(self.root / "sources.lock.json"),
            "annotation_sha256": {path.relative_to(self.root).as_posix(): sha256_file(path) for path in paths},
        })

    def test_counts_include_every_query_and_both_sides(self):
        inputs, evaluation, summary = build_records(self.root)
        self.assertEqual(len(inputs), 10)
        self.assertEqual([row["id"] for row in inputs], [row["id"] for row in evaluation])
        self.assertEqual(summary["scene_counts"], {
            "siuo_gen": 1, "siuo_mcqa": 1, "mssbench_chat": 1, "mssbench_embodied": 1,
        })
        self.assertEqual(summary["request_counts"], {
            "siuo_gen": 1, "siuo_mcqa": 1, "mssbench_chat": 4, "mssbench_embodied": 4,
        })
        self.assertEqual(summary["mssbench_pair_count"], 4)
        self.assertEqual(summary["unique_image_count"], 5)
        for pair_id in {row["pair_id"] for row in inputs if row["benchmark"] == "mssbench"}:
            pair = [row for row in inputs if row["pair_id"] == pair_id]
            self.assertEqual(len(pair), 2)
            self.assertNotEqual(pair[0]["media_paths"], pair[1]["media_paths"])

    def test_evaluation_fields_do_not_leak_into_model_inputs(self):
        inputs, evaluation, _ = build_records(self.root)
        excluded = {"category", "condition", "correct_option", "reference_answer", "safety_warning", "intent"}
        for row in inputs:
            self.assertFalse(excluded & row.keys())
        serialized = json.dumps(inputs)
        for marker in ("EVAL_WARNING", "EVAL_REFERENCE", "EVAL_INTENT", "EVAL_CAPTION", "EVAL_OBSERVATION"):
            self.assertNotIn(marker, serialized)
        self.assertEqual(evaluation[0]["reference_answer"], "EVAL_REFERENCE")
        self.assertEqual(evaluation[1]["correct_option"], "B")
        self.assertIn("options", inputs[1])

    def test_embodied_preserves_distinct_instruction_texts(self):
        inputs, _, _ = build_records(self.root)
        pair = [row for row in inputs if row["pair_id"] == "mssbench/embodied/0000/0"]
        self.assertEqual([row["query"] for row in pair], ["Move the blue item.", "Move the red item."])

    def test_unequal_instruction_lists_are_rejected(self):
        self.mss["embodied"][0]["unsafe_instructions"].pop()
        self.save_annotations()
        with self.assertRaisesRegex(ValueError, "Unequal instruction"):
            build_records(self.root)

    def test_duplicate_question_ids_are_rejected(self):
        self.gen.append(dict(self.gen[0]))
        self.save_annotations()
        with self.assertRaisesRegex(ValueError, "Duplicate SIUO"):
            build_records(self.root)

    def test_missing_image_is_rejected(self):
        (self.root / "raw/mssbench/chat/safe.png").unlink()
        with self.assertRaisesRegex(ValueError, "Missing image"):
            build_records(self.root)

    def test_corrupt_image_is_rejected(self):
        (self.root / "raw/mssbench/chat/safe.png").write_bytes(b"invalid-image")
        with self.assertRaisesRegex(ValueError, "Unreadable image"):
            build_records(self.root)

    def test_ambiguous_siuo_basename_is_rejected(self):
        self.image("raw/siuo/images/other/S-01.png", "black")
        with self.assertRaisesRegex(ValueError, "expected one match, found 2"):
            build_records(self.root)

    def test_identical_safe_and_unsafe_image_content_is_rejected(self):
        safe = self.root / "raw/mssbench/chat/safe.png"
        (safe.parent / "unsafe.png").write_bytes(safe.read_bytes())
        with self.assertRaisesRegex(ValueError, "Identical safe/unsafe"):
            build_records(self.root)

    def test_invalid_correct_option_is_rejected(self):
        self.mcqa[0]["correct_option"] = "Z"
        self.save_annotations()
        with self.assertRaisesRegex(ValueError, "correct_option"):
            build_records(self.root)

    def test_conflicting_mssbench_filenames_are_rejected(self):
        write_json_atomic(self.mss_path.with_name("combine.json"), {"different": True})
        with self.assertRaisesRegex(ValueError, "differ"):
            build_records(self.root)

    def test_legacy_mssbench_filename_is_supported(self):
        self.mss_path.rename(self.mss_path.with_name("combine.json"))
        inputs, _, summary = build_records(self.root)
        self.assertEqual(len(inputs), 10)
        self.assertIn("raw/mssbench/combine.json", summary["annotation_sha256"])

    def test_siuo_task_differences_are_reported_without_dropping_rows(self):
        self.mcqa[0]["question"] = "Choose the scene description."
        self.save_annotations()
        inputs, _, summary = build_records(self.root)
        self.assertEqual(len(inputs), 10)
        self.assertEqual(summary["siuo_task_overlap"]["different_inputs"], ["siuo/1001"])

    def test_media_paths_are_portable_and_cannot_escape_dataset(self):
        self.assertEqual(relative_media_path("nested\\image.png"), Path("nested/image.png"))
        for value in ("../image.png", "/tmp/image.png", "C:\\image.png", "nested/../../image.png", ""):
            with self.subTest(path=value), self.assertRaises(ValueError):
                relative_media_path(value)
        inputs, _, _ = build_records(self.root)
        for row in inputs:
            for value in row["media_paths"]:
                self.assertFalse(Path(value).is_absolute())
                self.assertNotIn("\\", value)

    def test_prepared_output_hashes_and_repeatability(self):
        first = prepare(self.root)
        second = prepare(self.root)
        self.assertEqual(first["provenance_status"], "manual_unlocked")
        self.assertEqual(first["output_sha256"], second["output_sha256"])
        for name, digest in first["output_sha256"].items():
            self.assertEqual(sha256_file(self.root / "prepared" / name), digest)

    def test_failed_validation_preserves_previous_outputs(self):
        prepare(self.root)
        paths = list((self.root / "prepared").iterdir())
        before = {path.name: path.read_bytes() for path in paths}
        self.gen.append(dict(self.gen[0]))
        self.save_annotations()
        with self.assertRaises(ValueError):
            prepare(self.root)
        self.assertEqual(before, {path.name: path.read_bytes() for path in paths})

    def test_locked_download_provenance_is_verified(self):
        self.save_provenance()
        manifest = prepare(self.root)
        self.assertEqual(manifest["provenance_status"], "locked")
        self.assertEqual(manifest["sources"], self.source_lock())

    def test_changed_annotations_cannot_keep_locked_provenance(self):
        self.save_provenance()
        prepare(self.root)
        before = (self.root / "prepared/manifest.json").read_bytes()
        self.gen[0]["reference_answer"] = "Modified reference"
        self.save_annotations()
        with self.assertRaisesRegex(ValueError, "Annotations differ"):
            prepare(self.root)
        self.assertEqual(before, (self.root / "prepared/manifest.json").read_bytes())

    def test_partial_download_cannot_be_prepared_as_locked(self):
        write_json_atomic(self.root / "sources.lock.json", self.source_lock())
        with self.assertRaisesRegex(ValueError, "Incomplete download provenance"):
            prepare(self.root)

    def test_changed_source_lock_is_rejected(self):
        self.save_provenance()
        lock = self.source_lock()
        lock["sources"]["mssbench"]["commit"] = "a" * 40
        write_json_atomic(self.root / "sources.lock.json", lock)
        with self.assertRaisesRegex(ValueError, "Source lock differs"):
            prepare(self.root)

    def test_output_cannot_overwrite_raw_data(self):
        with self.assertRaisesRegex(ValueError, "separate from raw"):
            prepare(self.root, self.root / "raw/mssbench")

    def test_existing_lock_reuses_revisions_without_network(self):
        path = self.root / "sources.lock.json"
        write_json_atomic(path, self.source_lock())
        api = Mock()
        with patch("scripts.download_datasets.resolve_github_commit") as git:
            actual = get_source_lock(path, dict.fromkeys(SOURCES), api)
        self.assertEqual(actual, self.source_lock())
        api.dataset_info.assert_not_called()
        git.assert_not_called()

    def test_existing_lock_rejects_revision_change(self):
        path = self.root / "sources.lock.json"
        write_json_atomic(path, self.source_lock())
        requests = dict.fromkeys(SOURCES)
        requests["mssbench"] = "other-branch"
        with self.assertRaisesRegex(ValueError, "new data root"):
            get_source_lock(path, requests, Mock())

    def test_download_pins_all_sources_and_records_mssbench_hash(self):
        module = types.ModuleType("huggingface_hub")
        module.HfApi = Mock(return_value=Mock(dataset_info=Mock(return_value=types.SimpleNamespace(sha="b" * 40))))
        module.snapshot_download = Mock()

        def fetch_fixture(url, destination, **kwargs):
            if destination.name == "UPSTREAM_README.md":
                destination.write_text("Fixture upstream information\n", encoding="utf-8")

        with patch.dict("sys.modules", {"huggingface_hub": module}), \
                patch("scripts.download_datasets.resolve_github_commit", return_value="a" * 40), \
                patch("scripts.download_datasets.fetch_text", side_effect=fetch_fixture) as fetch, \
                patch("scripts.download_datasets.importlib.metadata.version", return_value="fixture"):
            receipt = download(self.root, dict.fromkeys(SOURCES), 2)
        self.assertEqual(module.snapshot_download.call_count, 2)
        for call in module.snapshot_download.call_args_list:
            self.assertEqual(call.kwargs["revision"], "b" * 40)
        for call in fetch.call_args_list:
            self.assertIn("a" * 40, call.args[0])
        self.assertIn("raw/mssbench/combined.json", receipt["annotation_sha256"])
        self.assertEqual(prepare(self.root)["provenance_status"], "locked")

    def test_interrupted_download_keeps_lock_for_resume(self):
        module = types.ModuleType("huggingface_hub")
        module.HfApi = Mock(return_value=Mock(dataset_info=Mock(return_value=types.SimpleNamespace(sha="b" * 40))))
        module.snapshot_download = Mock()
        with patch.dict("sys.modules", {"huggingface_hub": module}), \
                patch("scripts.download_datasets.resolve_github_commit", return_value="a" * 40), \
                patch("scripts.download_datasets.fetch_text", side_effect=OSError("connection interrupted")):
            with self.assertRaisesRegex(OSError, "connection interrupted"):
                download(self.root, dict.fromkeys(SOURCES), 2)
        self.assertTrue((self.root / "sources.lock.json").is_file())
        self.assertFalse((self.root / "download_manifest.json").exists())
        module.snapshot_download.assert_not_called()


if __name__ == "__main__":
    unittest.main()
