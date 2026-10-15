"""Synthetic VLGuard integration; no download, weights or GPU execution."""

import contextlib
import copy
import io
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
from scripts.make_step1_demo import make_step1_demo
from scripts.make_vlguard_demo import make_vlguard_demo
from mm_sft_lab.common import digest, file_hash, read_json, read_jsonl, write_json
from mm_sft_lab.data_audit import audit_data
from mm_sft_lab.training_data import encode_training_record, load_train_config, train_plan
from mm_sft_lab.vlguard import prepare_vlguard, vlguard_rows
from test_step1 import FixtureProcessor



class VLGuardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.parent = Path(self.temp.name) / "step1"
        self.output = self.parent / "vlguard"
        self.quiet = contextlib.redirect_stdout(io.StringIO())
        self.quiet.__enter__()
        self.addCleanup(self.quiet.__exit__, None, None, None)
        make_step1_demo(self.parent)
        self.parent_hash = file_hash(self.parent / "prepared/manifest.json")
        self.manifest = make_vlguard_demo(self.parent)

    def test_both_safe_image_questions_are_independent_and_all_source_kinds_are_retained(self):
        rows = read_jsonl(self.output / "prepared/train_vlguard.jsonl")
        self.assertEqual(len(rows), 3)
        safe = [row for row in rows if row["source_id"] == "safe"]
        self.assertEqual(len(safe), 2)
        self.assertTrue(all(len(row["messages"]) == 2 for row in safe))
        self.assertEqual(safe[0]["media_paths"], safe[1]["media_paths"])
        self.assertEqual(set(self.manifest["audit"]["training_kind_counts"]),
                         {"unsafe_image", "safe_image_safe_instruction", "safe_image_unsafe_instruction"})

    def test_parent_snapshot_is_unchanged_and_counts_account_for_every_pair(self):
        self.assertEqual(file_hash(self.parent / "prepared/manifest.json"), self.parent_hash)
        audit = self.manifest["audit"]
        self.assertEqual(audit["raw_training_counts"]["vlguard"], 6)
        self.assertEqual(audit["excluded_counts"], {"duplicate_training_pair": 1, "evaluation_image_hash": 1, "evaluation_query_text": 1})
        self.assertEqual(audit["raw_training_counts"]["vlguard"], audit["training_counts"]["vlguard"] + sum(audit["excluded_counts"].values()))

    def test_vlguard_plan_and_mask_use_one_image_and_only_the_target(self):
        config = {**load_train_config(REPO / "configs/train/vlguard_lora.json"), "require_locked_data": False}
        self.assertEqual(train_plan(self.output, config)["training_records"], 3)
        row = read_jsonl(self.output / "prepared/train_vlguard.jsonl")[0]
        _, labels, info = encode_training_record(FixtureProcessor(), row, self.output, config)
        self.assertEqual([token for token in labels if token != -100], [ord(c) + 200 for c in row["messages"][-1]["content"]] + [9])
        self.assertEqual(info["images"], 1)
        with self.assertRaisesRegex(ValueError, "absent"):
            train_plan(self.parent, config)

    def test_repreparing_is_idempotent_and_changed_prepared_bytes_are_rejected(self):
        self.assertEqual(prepare_vlguard(self.parent, self.output, self.parent, allow_unlocked=True), self.manifest)
        (self.output / "prepared/train_vlguard.jsonl").write_text("{}\n")
        with self.assertRaisesRegex(ValueError, "snapshot changed"):
            prepare_vlguard(self.parent, self.output, self.parent, allow_unlocked=True)

    def test_changed_source_is_rejected_before_touching_frozen_images(self):
        row = read_jsonl(self.output / "prepared/train_vlguard.jsonl")[0]
        frozen_image = self.output / row["media_paths"][0]
        before = frozen_image.read_bytes()
        raw = self.parent / "raw/vlguard"
        with zipfile.ZipFile(raw / "train.zip", "w") as bundle:
            bundle.writestr("train/good/safe.png", b"changed image")
        source = read_json(raw / "download_manifest.json")
        source["files"]["train.zip"] = {"sha256": file_hash(raw / "train.zip"), "bytes": (raw / "train.zip").stat().st_size}
        write_json(raw / "download_manifest.json", source)
        with self.assertRaisesRegex(ValueError, "sources/evaluation would change"):
            prepare_vlguard(self.parent, self.output, self.parent, allow_unlocked=True)
        self.assertEqual(frozen_image.read_bytes(), before)

    def test_unknown_pair_kinds_and_empty_targets_are_rejected(self):
        rows = read_json(self.parent / "raw/vlguard/train.json")
        for change in ("wrong_kind", "empty_target", "duplicate_kind", "control_token"):
            bad = copy.deepcopy(rows)
            if change == "wrong_kind":
                bad[0]["instr-resp"][0] = {"instruction": "Wrong kind", "response": "Target"}
            elif change == "empty_target":
                bad[0]["instr-resp"][0]["response"] = " "
            elif change == "control_token":
                bad[0]["instr-resp"][0]["response"] = "<|im_end|>"
            else:
                bad[0]["instr-resp"][1] = copy.deepcopy(bad[0]["instr-resp"][0])
            with self.assertRaises(ValueError):
                list(vlguard_rows(bad))

    def test_reused_upstream_id_keeps_distinct_source_rows_and_targets(self):
        rows = read_json(self.parent / "raw/vlguard/train.json")[:1]
        repeated = copy.deepcopy(rows[0])
        repeated["instr-resp"][0]["safe_instruction"] = "A different synthetic question."
        rows.append(repeated)
        expanded = list(vlguard_rows(rows))
        self.assertEqual(len(expanded), 4)
        self.assertEqual(len({row["id"] for row in expanded}), 4)
        self.assertEqual({row["source_id"] for row in expanded}, {"safe"})
        self.assertEqual({row["source_index"] for row in expanded}, {0, 1})

    def test_output_cannot_overwrite_existing_snapshot_directories(self):
        for output in (self.parent, self.parent / "prepared/new", self.parent / "raw/new"):
            with self.assertRaisesRegex(ValueError, "separate|existing"):
                prepare_vlguard(self.parent, output, self.parent, allow_unlocked=True)

    def test_readonly_audit_checks_all_three_groups_without_changing_snapshots(self):
        before = file_hash(self.output / "prepared/manifest.json")
        result = audit_data(self.parent, REPO / "configs/train", vlguard_root=self.output)
        self.assertEqual(result["violations"], [])
        self.assertEqual(set(result["training"]), {"textvqa", "hh", "vlguard"})
        self.assertEqual(result["training"]["vlguard"]["training_samples"], 3)
        self.assertEqual(file_hash(self.output / "prepared/manifest.json"), before)
        self.assertEqual(file_hash(self.parent / "prepared/manifest.json"), self.parent_hash)

    def test_readonly_audit_detects_modified_exclusion_file(self):
        (self.output / "prepared/excluded_training.jsonl").write_text("{}\n")
        with self.assertRaisesRegex(ValueError, "output changed"):
            audit_data(self.parent, REPO / "configs/train", vlguard_root=self.output)


if __name__ == "__main__":
    unittest.main()
