"""Offline Step 1 contracts: no network, torch, model weights or API secrets."""

import contextlib
import gzip
import io
import json
import stat
import sys
import tempfile
import types
import unittest
import zipfile
from pathlib import Path
from unittest.mock import Mock, patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from scripts.make_step1_demo import make_step1_demo
from mm_sft_lab.backends import QwenBackend
from mm_sft_lab.benchmarks import messages_for, render_text
from mm_sft_lab.common import digest, file_hash, read_json, read_jsonl, write_json, write_jsonl
from mm_sft_lab.config import TASKS, load_config, load_judge_config
from mm_sft_lab.data import load_prepared
from mm_sft_lab.judge import judge_messages, judge_run, needs_judge, parse_labels
from mm_sft_lab.manual_judge import export_review, export_textvqa_test, import_manual
from mm_sft_lab.metrics import aggregate, score_run
from mm_sft_lab.preflight import adapter_identity, inference_identity
from mm_sft_lab.runner import generate
from mm_sft_lab.step1_data import hh_rows, hh_test_contexts, parse_hh, prepare_step1, wild_rows
from mm_sft_lab.step1_sources import TEXTVQA_URLS, download_source, extract_images, inside, verify_source
from mm_sft_lab.textvqa import normalize_answer, training_answer, vqa_accuracy
from mm_sft_lab.training import _audit_map, _verify_checkpoint, language_targets, lr_multiplier, run_training, trainable_report
from mm_sft_lab.training_data import (encode_training_record, load_train_config, load_training_records,
                                     supervised_labels, token_audit, train_plan, training_messages)


class FixtureProcessor:
    image_token_id = 99
    vision_start_token_id = 98
    vision_end_token_id = 97
    tokenizer = types.SimpleNamespace(pad_token_id=0)

    def apply_chat_template(self, messages, *, add_generation_prompt, **kwargs):
        ids, images = [], 0
        for message in messages:
            ids.append(1 if message["role"] == "user" else 2)
            for part in message["content"]:
                if part["type"] == "image":
                    ids.extend([98, 99, 97])
                    images += 1
                else:
                    ids.extend(ord(char) + 200 for char in part["text"])
            ids.append(9)
        if add_generation_prompt:
            ids.append(2)
        result = {"input_ids": [ids], "attention_mask": [[1] * len(ids)]}
        if images:
            result.update(pixel_values=[[1.0]], image_grid_thw=[[1, 2, 2]] * images)
        return result


class AnswerTests(unittest.TestCase):
    def test_vqa_soft_score_uses_leave_one_annotator_out(self):
        for matches, expected in ((0, 0), (1, 0.3), (2, 0.6), (3, 0.9), (4, 1), (10, 1)):
            self.assertAlmostEqual(vqa_accuracy("label", ["label"] * matches + ["other"] * (10 - matches)), expected)

    def test_answer_normalization_handles_numbers_articles_punctuation_and_decimals(self):
        self.assertEqual(normalize_answer("The TWO!"), "2")
        self.assertEqual(normalize_answer("1,000"), "1000")
        self.assertEqual(normalize_answer("3.14"), "3.14")
        self.assertEqual(normalize_answer("dont"), "don't")
        self.assertEqual(normalize_answer("cat's"), "cat 's")

    def test_missing_test_references_are_not_scored_as_zero_accuracy(self):
        for answers in ([], [""] * 10, ["a"]):
            with self.assertRaises(ValueError):
                vqa_accuracy("answer", answers)

    def test_majority_target_is_deterministic_and_preserves_original_spelling(self):
        self.assertEqual(training_answer(["Two"] * 4 + ["2"] * 3 + ["other"] * 3), "Two")
        self.assertEqual(training_answer(["B"] * 5 + ["C"] * 5), "B")

    def test_ocr_articles_and_symbols_remain_valid_training_targets(self):
        for target in ("a", "the", "a+", "A"):
            self.assertEqual(normalize_answer(target), "")
            self.assertEqual(training_answer([target] * 10), target)

    def test_empty_normalized_majority_is_not_discarded_in_favor_of_a_minority(self):
        self.assertEqual(training_answer(["a"] * 8 + ["other"] * 2), "a")
        self.assertEqual(training_answer([" "] * 4 + ["a+"] * 6), "a+")

    def test_raw_blank_targets_and_invalid_annotations_are_still_rejected(self):
        for answers in ([""] * 10, [" \t\n"] * 10):
            with self.assertRaisesRegex(ValueError, "no usable target"):
                training_answer(answers)
        for answers in (None, ["a"] * 9, ["a"] * 9 + [None]):
            with self.assertRaisesRegex(ValueError, "ten string annotations"):
                training_answer(answers)

    def test_hh_preserves_multiturn_context_and_only_validates_roles(self):
        rows = parse_hh("\n\nHuman: A\n\nAssistant: B\n\nHuman: C\n\nAssistant: D")
        self.assertEqual([row["content"] for row in rows], ["A", "B", "C", "D"])
        for value in ("Assistant: bad", "Human: A", "Human: A\n\nAssistant:", "Human: A\n\nHuman: B\n\nAssistant: C"):
            with self.assertRaises(ValueError):
                parse_hh(value)


class SourceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def hh_fixture(self, rows):
        path = self.root / "hh.jsonl.gz"
        with gzip.open(path, "wt", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row) + "\n")
        return path

    def test_hh_audited_mode_preserves_source_indices_and_rejects_mismatched_history(self):
        path = self.hh_fixture([
            {"chosen": "Human: A\n\nAssistant:", "rejected": "Human: A\n\nAssistant: Other"},
            {"chosen": "Human: B\n\nAssistant: Good", "rejected": "Human: C\n\nAssistant: Other"},
            {"chosen": "Human: D\n\nAssistant: Good", "rejected": "Human: D\n\nAssistant: Other"},
        ])
        with self.assertRaisesRegex(ValueError, "nonempty"):
            list(hh_rows(path))
        excluded = []
        rows = list(hh_rows(path, excluded=excluded))
        self.assertEqual([index for index, _ in rows], [2])
        self.assertEqual([row["reason"] for row in excluded], ["hh_invalid_conversation", "hh_history_mismatch"])
        self.assertEqual([row["line_number"] for row in excluded], [1, 2])
        self.assertEqual(len(rows) + len(excluded), 3)

    def test_hh_empty_test_target_still_protects_heldout_history(self):
        path = self.hh_fixture([{"chosen": "Human: Held out\n\nAssistant:",
                                 "rejected": "Human: Held out\n\nAssistant: Other"}])
        issues = []
        contexts, count = hh_test_contexts(path, issues=issues)
        self.assertEqual(count, 1)
        self.assertEqual(contexts, {digest([{"role": "user", "content": "held out"}])})
        self.assertEqual(issues, [])

    def test_unparseable_hh_test_history_is_recorded_per_branch(self):
        bad = "Human: A\n\nAssistant: B\n\nAssistant: C\n\nHuman: D\n\nAssistant: E"
        path = self.hh_fixture([{"chosen": bad, "rejected": bad}])
        issues = []
        contexts, count = hh_test_contexts(path, issues=issues)
        self.assertEqual((len(contexts), count), (0, 1))
        self.assertEqual([row["branch"] for row in issues], ["chosen", "rejected"])
        self.assertTrue(all(row["line_number"] == 1 for row in issues))

    def test_preparation_accounts_for_bad_hh_rows_and_empty_target_heldout_context(self):
        root = self.root / "audited"
        with patch("scripts.make_step1_demo.prepare_step1"):
            make_step1_demo(root)
        hh = root / "raw/hh/harmless-base"
        with gzip.open(hh / "train.jsonl.gz", "at", encoding="utf-8") as stream:
            for row in (
                {"chosen": "Human: Bad\n\nAssistant:", "rejected": "Human: Bad\n\nAssistant: Other"},
                {"chosen": "Human: A\n\nAssistant: Yes", "rejected": "Human: B\n\nAssistant: No"},
            ):
                stream.write(json.dumps(row) + "\n")
        with gzip.open(hh / "test.jsonl.gz", "at", encoding="utf-8") as stream:
            stream.write(json.dumps({"chosen": "Human: Tell me a neutral greeting.\n\nAssistant:",
                                     "rejected": "Human: Tell me a neutral greeting.\n\nAssistant: Other"}) + "\n")
        manifest_path = root / "raw/hh/download_manifest.json"
        manifest = read_json(manifest_path)
        for split in ("train", "test"):
            path = hh / f"{split}.jsonl.gz"
            manifest["files"][f"harmless-base/{split}.jsonl.gz"] = {"sha256": file_hash(path), "bytes": path.stat().st_size}
        write_json(manifest_path, manifest)
        result = prepare_step1(root, root / "legacy", allow_unlocked=True)
        audit = result["audit"]
        self.assertEqual(audit["raw_training_counts"]["hh"], 7)
        self.assertEqual(audit["hh_format_valid_count"], 5)
        self.assertEqual(audit["hh_format_excluded_count"], 2)
        self.assertEqual(audit["training_counts"]["hh"], 1)
        excluded = [row for row in read_jsonl(root / "prepared/excluded_training.jsonl") if row["id"].startswith("train/hh/")]
        self.assertEqual(len(excluded) + audit["training_counts"]["hh"], 7)
        self.assertTrue(any(row["id"] == "train/hh/000005" and row["line_number"] == 6 for row in excluded))
        self.assertEqual(audit["excluded_counts"]["hh_test_context"], 3)
        self.assertEqual(audit["hh_test_context_issue_count"], 0)
        self.assertIn("hh_test_context_issues.jsonl", result["output_sha256"])
        self.assertEqual(prepare_step1(root, root / "legacy", allow_unlocked=True), result)

    def test_preparation_preserves_raw_ocr_target_when_normalization_is_empty(self):
        root = self.root / "ocr-letter"
        with patch("scripts.make_step1_demo.prepare_step1"):
            make_step1_demo(root)
        path = root / "raw/textvqa/TextVQA_0.5.1_train.json"
        annotations = read_json(path)
        annotations["data"][0]["answers"] = ["A+"] * 10
        write_json(path, annotations)
        manifest_path = root / "raw/textvqa/download_manifest.json"
        manifest = read_json(manifest_path)
        manifest["files"][path.name] = {"sha256": file_hash(path), "bytes": path.stat().st_size}
        write_json(manifest_path, manifest)
        result = prepare_step1(root, root / "legacy", allow_unlocked=True)
        rows = read_jsonl(root / "prepared/train_textvqa.jsonl")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["messages"][-1]["content"], "A+")
        config = {**load_train_config(REPO / "configs/train/textvqa_lora.json"), "require_locked_data": False}
        _, labels, summary = encode_training_record(FixtureProcessor(), rows[0], root, config)
        self.assertEqual([token for token in labels if token != -100], [ord(char) + 200 for char in "A+"] + [9])
        self.assertGreater(summary["supervised_tokens"], 0)
        self.assertTrue(result["target_policy"]["textvqa"].endswith("v2"))

    def test_path_validation_rejects_windows_and_posix_escapes(self):
        for name in ("../x", "/x", "C:/x", "a\\b"):
            with self.assertRaises(ValueError):
                inside(self.root, name)

    def test_zip_traversal_and_symlinks_are_rejected_before_writing(self):
        for name in ("../escaped.txt", "C:/escaped.txt"):
            path = self.root / "bad.zip"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr(name, "fixture")
            with self.assertRaises(ValueError):
                extract_images(path, self.root / "out")
        path = self.root / "link.zip"
        with zipfile.ZipFile(path, "w") as archive:
            entry = zipfile.ZipInfo("link")
            entry.create_system = 3
            entry.external_attr = (stat.S_IFLNK | 0o777) << 16
            archive.writestr(entry, "../elsewhere")
        with self.assertRaisesRegex(ValueError, "symlink"):
            extract_images(path, self.root / "out")

    def test_textvqa_download_resumes_completed_assets_and_verifies_changes(self):
        calls = []
        def fetch(url, path):
            calls.append(path.name)
            if len(calls) == 2:
                raise TimeoutError("fixture interruption")
            path.write_text(url, encoding="utf-8")
        with self.assertRaises(TimeoutError):
            download_source(self.root, "textvqa", fetch=fetch)
        result = download_source(self.root, "textvqa", fetch=fetch)
        self.assertEqual(len(result["files"]), len(TEXTVQA_URLS))
        self.assertEqual(calls.count("TextVQA_0.5.1_train.json"), 1)
        download_source(self.root, "textvqa", fetch=Mock(side_effect=AssertionError("network")))
        (self.root / "raw/textvqa/TextVQA_0.5.1_train.json").write_text("changed")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            verify_source(self.root, "textvqa")

    def test_hh_download_pins_and_selects_only_harmless(self):
        api = types.SimpleNamespace(dataset_info=Mock(return_value=types.SimpleNamespace(sha="a" * 40)))
        captured = []
        def snapshot(**kwargs):
            captured.append(kwargs)
            base = Path(kwargs["local_dir"]) / "harmless-base"
            base.mkdir(parents=True)
            for split in ("train", "test"):
                (base / f"{split}.jsonl.gz").write_bytes(b"fixture")
        download_source(self.root, "hh", hf_api=api, snapshot=snapshot)
        self.assertEqual(captured[0]["revision"], "a" * 40)
        self.assertEqual(captured[0]["allow_patterns"], ["harmless-base/*.jsonl.gz", "README.md"])
        download_source(self.root, "hh", hf_api=Mock(side_effect=AssertionError("network")), snapshot=snapshot)
        with self.assertRaisesRegex(ValueError, "Revision changed"):
            download_source(self.root, "hh", revision="different", hf_api=api, snapshot=snapshot)

    def test_vlguard_download_pins_only_training_assets_and_reuses_verified_files(self):
        api = types.SimpleNamespace(dataset_info=Mock(return_value=types.SimpleNamespace(sha="c" * 40)))
        captured = []
        def snapshot(**kwargs):
            captured.append(kwargs)
            base = Path(kwargs["local_dir"])
            (base / "train.json").write_text("[]", encoding="utf-8")
            with zipfile.ZipFile(base / "train.zip", "w") as archive:
                archive.writestr("fixture.jpg", b"fixture image")
            (base / "README.md").write_text("fixture documentation", encoding="utf-8")
        result = download_source(self.root, "vlguard", hf_api=api, snapshot=snapshot)
        self.assertEqual(captured[0]["repo_id"], "ys-zong/VLGuard")
        self.assertEqual(captured[0]["revision"], "c" * 40)
        self.assertEqual(captured[0]["allow_patterns"], ["train.json", "train.zip", "README.md"])
        self.assertEqual(set(result["files"]), {"train.json", "train.zip", "README.md"})
        self.assertEqual(verify_source(self.root, "vlguard"), result)
        no_network = Mock(side_effect=AssertionError("network"))
        self.assertEqual(download_source(self.root, "vlguard", hf_api=no_network, snapshot=no_network), result)
        no_network.assert_not_called()
        (self.root / "raw/vlguard/train.zip").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            verify_source(self.root, "vlguard")

    def test_vlguard_missing_images_retries_at_the_original_commit(self):
        api = types.SimpleNamespace(dataset_info=Mock(return_value=types.SimpleNamespace(sha="d" * 40)))
        base = self.root / "raw/vlguard"
        def incomplete(**kwargs):
            (base / "train.json").write_text("[]", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "missing required assets"):
            download_source(self.root, "vlguard", hf_api=api, snapshot=incomplete)
        self.assertFalse((base / "download_manifest.json").exists())
        self.assertEqual(read_json(base / "source.lock.json")["commit"], "d" * 40)
        revisions = []
        def complete(**kwargs):
            revisions.append(kwargs["revision"])
            (base / "train.zip").write_bytes(b"fixture archive")
        result = download_source(self.root, "vlguard", hf_api=api, snapshot=complete)
        self.assertEqual(revisions, ["d" * 40])
        api.dataset_info.assert_called_once()
        self.assertEqual(set(result["files"]), {"train.json", "train.zip"})

    def test_vlguard_manifest_must_include_training_images(self):
        base = self.root / "raw/vlguard"
        base.mkdir(parents=True)
        lock = {"repo_id": "ys-zong/VLGuard", "commit": "e" * 40}
        write_json(base / "source.lock.json", lock)
        write_json(base / "download_manifest.json", {
            "schema_version": 1, "source": "vlguard", "provenance": "locked",
            "source_lock_sha256": digest(lock), "files": {"train.json": {"sha256": "fixture"}},
        })
        with self.assertRaisesRegex(ValueError, "missing required assets"):
            verify_source(self.root, "vlguard")

    def test_wild_eval_parser_keeps_quotes_and_newlines(self):
        import csv
        path = self.root / "eval.tsv"
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["adversarial", "data_type"], delimiter="\t")
            writer.writeheader()
            writer.writerow({"adversarial": 'A "quoted"\nexample', "data_type": "adversarial_benign"})
        self.assertEqual(list(wild_rows(path))[0][2], 'A "quoted"\nexample')


class Step1Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "step1"
        self.manifest = make_step1_demo(self.root)
        self.config = {**load_config(REPO / "configs/step1_full.json"), "backend": "mock", "require_locked_data": False}
        self.train_config = {**load_train_config(REPO / "configs/train/textvqa_lora.json"), "require_locked_data": False}
        self.run = self.root / "run"
        self.quiet = contextlib.redirect_stdout(io.StringIO())
        self.quiet.__enter__()
        self.addCleanup(self.quiet.__exit__, None, None, None)

    def generated(self, *, include_test=False):
        config = {**self.config, "tasks": list(TASKS)} if include_test else self.config
        return generate(config, self.root, self.run)

    def test_conversion_counts_and_all_filter_reasons_are_visible(self):
        self.assertEqual(self.manifest["audit"]["training_counts"], {"textvqa": 1, "hh": 2})
        for reason in ("heldout_image_id", "evaluation_image_hash", "duplicate_training_pair", "hh_test_context",
                       "evaluation_query_text", "duplicate_chosen_conversation"):
            self.assertEqual(self.manifest["audit"]["excluded_counts"][reason], 1)

    def test_repreparing_identical_data_preserves_manifest_and_hashes(self):
        before = file_hash(self.root / "prepared/manifest.json")
        result = prepare_step1(self.root, self.root / "legacy", allow_unlocked=True)
        self.assertEqual(result, self.manifest)
        self.assertEqual(before, file_hash(self.root / "prepared/manifest.json"))

    def test_changed_prepared_data_is_not_silently_overwritten(self):
        path = self.root / "prepared/train_hh.jsonl"
        path.write_text("{}\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "snapshot changed"):
            prepare_step1(self.root, self.root / "legacy", allow_unlocked=True)

    def test_text_only_messages_have_no_placeholder_image_or_reference(self):
        records, labels, _ = load_prepared(self.root, self.config)
        for row in records:
            if row["benchmark"] == "wildjailbreak":
                payload = messages_for(row, self.root)
                self.assertEqual(payload[0]["content"], [{"type": "text", "text": row["query"]}])
            if row["benchmark"] == "textvqa":
                self.assertNotIn("Label", render_text(row))

    def test_fake_data_cannot_start_gpu_training_even_with_unlocked_config(self):
        with patch("mm_sft_lab.training.training_preflight", side_effect=AssertionError("GPU")):
            with self.assertRaisesRegex(ValueError, "Synthetic fixtures"):
                run_training(self.root, self.train_config, self.root / "train", phase="smoke")

    def test_training_plan_does_not_import_model_dependencies(self):
        self.assertEqual(train_plan(self.root, self.train_config)["optimizer_steps"], 1)
        real = load_train_config(REPO / "configs/train/textvqa_lora.json")
        with self.assertRaisesRegex(ValueError, "locked"):
            train_plan(self.root, real)

    def test_training_file_and_image_hashes_are_enforced(self):
        records, _ = load_training_records(self.root, self.train_config)
        image = self.root / records[0]["media_paths"][0]
        image.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "Training image changed"):
            load_training_records(self.root, self.train_config)

    def test_hh_history_is_context_only_and_last_target_is_supervised(self):
        config = {**self.train_config, "dataset": "hh"}
        records, _ = load_training_records(self.root, config)
        record = next(row for row in records if len(row["messages"]) == 4)
        full, labels, summary = encode_training_record(FixtureProcessor(), record, self.root, config)
        target = [token for token in labels if token != -100]
        self.assertEqual(target, [ord(char) + 200 for char in "Goodbye."] + [9])
        self.assertEqual(summary["supervised_tokens"], len(target))
        self.assertNotIn("pixel_values", full)

    def test_image_and_user_tokens_are_masked(self):
        records, _ = load_training_records(self.root, self.train_config)
        full, labels, info = encode_training_record(FixtureProcessor(), records[0], self.root, self.train_config)
        ids = full["input_ids"][0]
        for index, token in enumerate(ids):
            if token in (98, 99, 97):
                self.assertEqual(labels[index], -100)
        self.assertEqual(info["images"], 1)
        self.assertEqual([value for value in labels if value != -100], [ord(char) + 200 for char in "Label"] + [9])

    def test_overlength_audit_records_failure_without_dropping_samples(self):
        result = token_audit(self.root, {**self.train_config, "max_tokens": 4}, self.root / "audit", processor=FixtureProcessor())
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["errors"], 1)
        self.assertEqual(len(read_jsonl(self.root / "audit/errors.jsonl")), 1)

    def test_full_token_audit_is_marked_mock(self):
        result = token_audit(self.root, self.train_config, self.root / "audit", processor=FixtureProcessor())
        self.assertTrue(result["full_dataset"])
        self.assertTrue(result["is_mock"])
        self.assertGreater(result["supervised_tokens"], 0)

    def test_no_api_is_required_for_textvqa_val_scoring(self):
        self.generated()
        metrics, _ = score_run(self.run)
        self.assertTrue(metrics["tasks"]["textvqa/val"]["complete"])
        self.assertEqual(metrics["tasks"]["textvqa/val"]["accuracy"], 0)
        self.assertIsNone(metrics["tasks"]["wildjailbreak/harmful"]["asr"])

    def test_api_judge_skips_all_textvqa_tasks(self):
        self.generated(include_test=True)
        result = judge_run(self.run, self.root / "judge", load_judge_config(REPO / "configs/judge.example.json"), mock=True)
        self.assertEqual(result["expected_requests"], 7)
        self.assertEqual(result["successful_requests"], 7)
        rows = read_jsonl(self.root / "judge/judgments.jsonl")
        self.assertFalse(any(row["id"].startswith("textvqa/") for row in rows))

    def test_wild_judge_includes_query_but_no_gold_condition_or_image(self):
        records, labels, _ = load_prepared(self.root, self.config)
        record = next(row for row in records if row["benchmark"] == "wildjailbreak")
        label = {row["id"]: row for row in labels}[record["id"]]
        payload = judge_messages(record, label, "a quoted answer", self.root)
        self.assertEqual(len(payload[0]["content"]), 1)
        self.assertIn(record["query"], payload[0]["content"][0]["text"])
        self.assertNotIn('"condition"', payload[0]["content"][0]["text"])

    def test_qwen_text_only_generation_does_not_require_fake_visual_inputs(self):
        class Tokens:
            def __init__(self, values): self.values = values
            def __getitem__(self, key): return Tokens(self.values[key])
            def tolist(self): return self.values
            def sum(self): return types.SimpleNamespace(item=lambda: sum(self.values))
        class Batch(dict):
            def to(self, device): return self
        inputs = Batch(input_ids=types.SimpleNamespace(shape=(1, 3)), attention_mask=[Tokens([1, 1, 1])])
        backend = object.__new__(QwenBackend)
        backend.torch = types.SimpleNamespace(cuda=types.SimpleNamespace(
            synchronize=lambda: None, reset_peak_memory_stats=lambda device: None,
            max_memory_allocated=lambda device: 100, max_memory_reserved=lambda device: 200), inference_mode=contextlib.nullcontext)
        backend.processor = Mock()
        backend.processor.tokenizer.pad_token_id = 0
        backend.processor.apply_chat_template.return_value = inputs
        backend.processor.decode.return_value = "text-only completion"
        backend.model = Mock()
        backend.model.generation_config.eos_token_id = 9
        backend.model.generate.return_value = [Tokens([1, 2, 3, 4, 9])]
        records, _, _ = load_prepared(self.root, self.config)
        row = next(record for record in records if record["benchmark"] == "wildjailbreak")
        module = types.ModuleType("transformers")
        module.set_seed = Mock()
        with patch.dict("sys.modules", {"transformers": module}):
            result = backend.generate([row], self.root, self.config, [42])[0]
        self.assertFalse(result["pixel_values_present"])
        self.assertEqual(result["image_count"], 0)
        self.assertEqual(result["image_grid_thw"], [])
        self.assertEqual(result["output_tokens"], 2)

    def test_manual_review_import_scores_and_keeps_mock_marker(self):
        self.generated()
        source = self.root / "review.jsonl"
        export_review(self.run, source)
        rows = read_jsonl(source)
        self.assertEqual(len(rows), 7)
        for row in rows:
            row["labels"] = ({"safety": True, "effectiveness": "very_effective"} if row["benchmark"] == "siuo" else
                             {"class_index": 2} if row["benchmark"] == "mssbench" else
                             {"harmful": row["task"] == "harmful", "refusal": False})
        write_jsonl(source, rows)
        result = import_manual(self.run, source, self.root / "manual")
        self.assertEqual(result["status"], "complete")
        metrics, _ = score_run(self.run, self.root / "manual")
        self.assertTrue(metrics["complete"])
        self.assertTrue(metrics["is_mock"])
        self.assertEqual(metrics["tasks"]["wildjailbreak/harmful"]["asr"], 1)

    def test_null_manual_labels_remain_pending(self):
        self.generated()
        source = self.root / "review.jsonl"
        export_review(self.run, source)
        rows = read_jsonl(source)
        rows[0]["labels"] = {"safety": True, "effectiveness": "very_effective"}
        write_jsonl(source, rows)
        result = import_manual(self.run, source, self.root / "manual")
        self.assertEqual(result["pending_requests"], 6)

    def test_manual_response_edits_are_rejected(self):
        self.generated()
        source = self.root / "review.jsonl"
        export_review(self.run, source)
        rows = read_jsonl(source)
        rows[0]["response"] = "edited"
        write_jsonl(source, rows)
        with self.assertRaisesRegex(ValueError, "Edit labels only"):
            import_manual(self.run, source, self.root / "manual")

    def test_textvqa_test_export_does_not_invent_an_accuracy(self):
        self.generated(include_test=True)
        output = self.root / "test_submission.json"
        result = export_textvqa_test(self.run, output)
        self.assertEqual(result["official_score"], "pending")
        self.assertEqual(read_json(output)[0]["question_id"], 20)
        metrics, _ = score_run(self.run)
        self.assertIsNone(metrics["tasks"]["textvqa/test"]["accuracy"])

    def test_shipped_recipes_differ_only_in_training_dataset(self):
        first = load_train_config(REPO / "configs/train/textvqa_lora.json")
        second = load_train_config(REPO / "configs/train/hh_lora.json")
        self.assertEqual({k: v for k, v in first.items() if k != "dataset"}, {k: v for k, v in second.items() if k != "dataset"})
        for name in ("step1_smoke", "step1_full", "textvqa_test"):
            load_config(REPO / f"configs/{name}.json")


class TrainingContractTests(unittest.TestCase):
    def test_last_answer_mask_rejects_template_mismatch_empty_target_and_overlength(self):
        self.assertEqual(supervised_labels([1, 2, 3, 4], [1, 2], 4), [-100, -100, 3, 4])
        for full, prompt, maximum in (([1, 2], [1, 3], 4), ([1, 2], [1, 2], 4), ([1, 2, 3], [1], 2)):
            with self.assertRaises(ValueError):
                supervised_labels(full, prompt, maximum)

    def test_lora_target_discovery_excludes_visual_and_other_language_modules(self):
        good = [f"model.language_model.layers.0.self_attn.{name}" for name in ("q_proj", "k_proj", "v_proj", "o_proj")]
        names = good + ["model.visual.blocks.0.q_proj", "model.visual.merger.q_proj", "model.language_model.embed_tokens"]
        self.assertEqual(language_targets(names, ["q_proj", "k_proj", "v_proj", "o_proj"]), sorted(good))

    def test_trainable_report_rejects_visual_and_unfrozen_base_parameters(self):
        parameter = types.SimpleNamespace(requires_grad=True, numel=lambda: 4, dtype="float32")
        for name in ("model.visual.merger.weight", "model.language_model.layers.0.self_attn.q_proj.weight"):
            with self.assertRaisesRegex(ValueError, "Unexpected trainable"):
                trainable_report(types.SimpleNamespace(named_parameters=lambda: [(name, parameter)]))
        good = "base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A.default.weight"
        self.assertEqual(trainable_report(types.SimpleNamespace(named_parameters=lambda: [(good, parameter)]))["trainable_parameters"], 4)

    def test_scheduler_handles_warmup_and_small_datasets(self):
        self.assertEqual(lr_multiplier(0, 2, 0, "linear"), 1)
        self.assertEqual(lr_multiplier(0, 10, 2, "linear"), 0)
        self.assertEqual(lr_multiplier(2, 10, 2, "linear"), 1)
        self.assertEqual(lr_multiplier(10, 10, 2, "cosine"), 0)

    def test_training_config_rejects_batch_mismatch_visual_training_and_typos(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.json"
            base = read_json(REPO / "configs/train/textvqa_lora.json")
            for changes in ({"micro_batch_size": 2}, {"freeze_visual": False}, {"max_token": 4096}, {"dataset": "unknown"}):
                write_json(path, {**base, **changes})
                with self.assertRaises(ValueError):
                    load_train_config(path)

    def test_checkpoint_identity_and_file_hash_are_verified_without_torch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            state = root / "training_state.pt"
            state.write_bytes(b"fixture-state")
            (root / "adapter").mkdir()
            (root / "adapter/adapter_config.json").write_text("{}")
            (root / "adapter/adapter_model.safetensors").write_bytes(b"fixture-weights")
            files = [state, root / "adapter/adapter_config.json", root / "adapter/adapter_model.safetensors"]
            write_json(root / "checkpoint.json", {"run_fingerprint": "recipe", "files_sha256": {p.relative_to(root).as_posix(): file_hash(p) for p in files}})
            _verify_checkpoint(root, "recipe")
            with self.assertRaisesRegex(ValueError, "different training"):
                _verify_checkpoint(root, "other")
            state.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "Checkpoint file changed"):
                _verify_checkpoint(root, "recipe")

    def test_token_audit_rejects_code_or_dependency_drift_before_loading_weights(self):
        with tempfile.TemporaryDirectory() as temp:
            root, audit = Path(temp), Path(temp) / "audit"
            write_json(root / "prepared/manifest.json", {"fixture": True})
            entries = [{"id": "one", "input_tokens": 10}]
            write_jsonl(audit / "tokens.jsonl", entries)
            contract = {"status": "passed", "full_dataset": True, "is_mock": False,
                        "config_sha256": digest({}), "data_manifest_sha256": file_hash(root / "prepared/manifest.json"),
                        "model_fingerprint": "base", "tokens_sha256": file_hash(audit / "tokens.jsonl"),
                        "code_sha256": "code", "tokenization_environment": {"python": "fixture"}}
            with patch("mm_sft_lab.training.code_identity", return_value={"package_sha256": "code"}), \
                    patch("mm_sft_lab.training.tokenization_environment", return_value={"python": "fixture"}):
                write_json(audit / "manifest.json", contract)
                self.assertEqual(_audit_map(audit, {}, root, entries, "base", full=True), {"one": entries[0]})
                for key, value in (("code_sha256", "other-code"), ("tokenization_environment", {"python": "other-version"})):
                    write_json(audit / "manifest.json", {**contract, key: value})
                    with self.assertRaisesRegex(ValueError, "differs"):
                        _audit_map(audit, {}, root, entries, "base", full=True)

    def test_adapter_weights_affect_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_json(root / "adapter_config.json", {"peft_type": "LORA"})
            weights = root / "adapter_model.safetensors"
            weights.write_bytes(b"first")
            first = adapter_identity(root)["fingerprint"]
            weights.write_bytes(b"second")
            self.assertNotEqual(adapter_identity(root)["fingerprint"], first)

    def test_adapter_base_provenance_is_required_and_verified(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_json(root / "adapter_config.json", {"peft_type": "LORA"})
            (root / "adapter_model.safetensors").write_bytes(b"fixture")
            config = {"model_path": "fixture-base", "adapter_path": str(root)}
            with patch("mm_sft_lab.preflight.model_identity", return_value={"fingerprint": "base-A"}):
                with self.assertRaisesRegex(ValueError, "provenance"):
                    inference_identity(config)
                write_json(root / "training_provenance.json", {"base_fingerprint": "base-B"})
                with self.assertRaisesRegex(ValueError, "different base"):
                    inference_identity(config)
                write_json(root / "training_provenance.json", {"base_fingerprint": "base-A"})
                self.assertIn("adapter", inference_identity(config))


if __name__ == "__main__":
    unittest.main()
