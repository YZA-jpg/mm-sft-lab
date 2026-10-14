"""Offline tests for experiment identity, recovery, API payloads, and metrics."""

import base64
import contextlib
import copy
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scripts.make_demo import make_demo
from mm_sft_lab.backends import ChatClient, MockBackend, QwenBackend, api_messages, trim_completion
from mm_sft_lab.benchmarks import messages_for, parse_mcqa, render_text
from mm_sft_lab.cli import main
from mm_sft_lab.common import append_jsonl, digest, latest_attempts, read_json, read_jsonl, stage_lock, write_json, write_jsonl
from mm_sft_lab.config import load_config, load_judge_config
from mm_sft_lab.data import load_prepared, select_records
from mm_sft_lab.judge import judge_messages, judge_run, parse_labels
from mm_sft_lab.metrics import aggregate, compare_runs, score_run
from mm_sft_lab.preflight import model_identity
from mm_sft_lab.runner import generate, load_run, prediction_map


REPO = Path(__file__).resolve().parents[1]


class FrameworkTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.data = self.root / "data"
        self.run = self.root / "run"
        make_demo(self.data)
        self.config = load_config(REPO / "configs/mock.json")
        self.judge_config = load_judge_config(REPO / "configs/judge.example.json")
        self.quiet = contextlib.redirect_stdout(io.StringIO())
        self.quiet.__enter__()
        self.addCleanup(self.quiet.__exit__, None, None, None)

    def config_file(self, changes):
        path = self.root / "config.json"
        write_json(path, {**self.config, **changes})
        return path

    def generated(self, **kwargs):
        return generate(self.config, self.data, self.run, **kwargs)

    def test_unknown_config_keys_cannot_silently_change_protocol(self):
        with self.assertRaisesRegex(ValueError, "Unknown"):
            load_config(self.config_file({"max_new_token": 10}))

    def test_duplicate_greedy_responses_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "greedy copies"):
            load_config(self.config_file({"seeds": [42, 43, 44]}))

    def test_sampling_requires_single_request_batches(self):
        with self.assertRaisesRegex(ValueError, "batch_size=1"):
            load_config(self.config_file({"do_sample": True, "batch_size": 4}))

    def test_even_sampling_votes_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "odd number"):
            load_config(self.config_file({"do_sample": True, "seeds": [42, 43], "tasks": ["siuo/gen"]}))

    def test_shipped_configs_are_valid(self):
        for name in ("baseline_smoke", "baseline_full", "siuo_3sample", "model_api.example", "mock"):
            load_config(REPO / f"configs/{name}.json")

    def test_official_run_requires_locked_data(self):
        config = load_config(REPO / "configs/baseline_smoke.json")
        with self.assertRaisesRegex(ValueError, "locked official data"):
            load_prepared(self.data, config)

    def test_prepared_file_edits_are_detected(self):
        path = self.data / "prepared/inputs.jsonl"
        path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            load_prepared(self.data, self.config)

    def test_image_edits_are_detected(self):
        (self.data / "raw/siuo/images/S-01.png").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "Image hash mismatch"):
            load_prepared(self.data, self.config)

    def test_smoke_selection_keeps_whole_pairs(self):
        records, _, _ = load_prepared(self.data, self.config)
        extra = []
        for row in records:
            if row["benchmark"] == "mssbench":
                extra.append({**row, "id": row["id"] + "extra", "pair_id": row["pair_id"] + "extra"})
        selected = select_records(records + extra, self.config["tasks"], 1)
        self.assertEqual(len(selected), 6)
        self.assertEqual(len({row["pair_id"] for row in selected if row["pair_id"]}), 2)

    def test_only_whitelisted_content_is_rendered_for_model(self):
        records, _, _ = load_prepared(self.data, self.config)
        for row in records:
            contaminated = {**row, "reference_answer": "SECRET_REFERENCE", "safety_warning": "SECRET_WARNING", "condition": "SECRET_CONDITION"}
            payload = json.dumps(messages_for(contaminated, self.data))
            for sentinel in ("SECRET_REFERENCE", "SECRET_WARNING", "SECRET_CONDITION"):
                self.assertNotIn(sentinel, payload)
            self.assertNotIn('"role": "system"', payload)
        self.assertIn("Available actions", render_text(records[-1]))

    def test_mcqa_extraction_does_not_guess_ambiguous_answers(self):
        options = {"A": "first", "B": "second"}
        for text in ("A", "(A)", "Answer: A", "**A**", "A."):
            self.assertEqual(parse_mcqa(text, options), "A")
        for text in ("A or B", "I prefer A because...", "", "C"):
            self.assertIsNone(parse_mcqa(text, options))

    def test_api_images_include_bytes_and_correct_mime(self):
        path = self.data / "raw/siuo/images/space %20 name.png"
        original = self.data / "raw/siuo/images/S-01.png"
        path.write_bytes(original.read_bytes())
        payload = api_messages([{"role": "user", "content": [{"type": "image", "image": path.as_uri()}]}])
        url = payload[0]["content"][0]["image_url"]["url"]
        self.assertTrue(url.startswith("data:image/png;base64,"))
        self.assertEqual(base64.b64decode(url.split(",", 1)[1]), original.read_bytes())

    def test_api_client_uses_env_and_separate_token_parameter(self):
        module = types.ModuleType("openai")
        response = types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="hello"), finish_reason="stop")],
            usage=types.SimpleNamespace(prompt_tokens=4, completion_tokens=2), id="call-1", model="served-model", system_fingerprint="fp-1")
        client = Mock()
        client.chat.completions.create.return_value = response
        module.OpenAI = Mock(return_value=client)
        config = {**self.config["api"], "token_parameter": "max_completion_tokens"}
        with patch.dict("sys.modules", {"openai": module}), patch.dict("os.environ", {
            "MODEL_API_KEY": "secret-fixture", "MODEL_BASE_URL": "https://example.invalid/v1", "MODEL_ID": "model-fixture"
        }):
            adapter = ChatClient(config)
            result = adapter.complete([{"role": "user", "content": [{"type": "text", "text": "hi"}]}], max_tokens=7, temperature=0, seed=42)
        kwargs = client.chat.completions.create.call_args.kwargs
        self.assertEqual(kwargs["max_completion_tokens"], 7)
        self.assertNotIn("seed", kwargs)
        self.assertEqual(result["response"], "hello")
        self.assertNotIn("secret-fixture", json.dumps(adapter.identity))
        self.assertNotIn("secret-fixture", adapter.safe_error(ValueError("secret-fixture")))

    def test_completion_trimming_handles_eos_and_limit(self):
        self.assertEqual(trim_completion([4, 5, 9, 0, 0], {9}, 0, 5), ([4, 5, 9], "stop"))
        self.assertEqual(trim_completion([4, 5], {9}, 0, 2), ([4, 5], "length"))

    def test_qwen_batch_decode_slices_padded_width_not_attention_length(self):
        class Tokens:
            def __init__(self, values): self.values = values
            def __getitem__(self, item): return Tokens(self.values[item])
            def tolist(self): return self.values
            def sum(self): return types.SimpleNamespace(item=lambda: sum(self.values))
        class Batch(dict):
            def to(self, device): return self
        inputs = Batch(input_ids=types.SimpleNamespace(shape=(2, 5)),
                       attention_mask=[Tokens([0, 0, 1, 1, 1]), Tokens([1, 1, 1, 1, 1])],
                       image_grid_thw=Tokens([[1, 2, 2], [1, 4, 2]]),
                       pixel_values=types.SimpleNamespace(numel=lambda: 3))
        backend = object.__new__(QwenBackend)
        backend.torch = types.SimpleNamespace(cuda=types.SimpleNamespace(
            synchronize=lambda: None, reset_peak_memory_stats=lambda device: None,
            max_memory_allocated=lambda device: 1024, max_memory_reserved=lambda device: 2048,
        ), inference_mode=contextlib.nullcontext)
        backend.processor = Mock()
        backend.processor.tokenizer.pad_token_id = 0
        backend.processor.apply_chat_template.return_value = inputs
        backend.processor.decode.side_effect = lambda tokens, **kwargs: str(tokens)
        backend.model = Mock()
        backend.model.generation_config.eos_token_id = 9
        backend.model.generate.return_value = [Tokens([0, 0, 1, 2, 3, 4, 9, 0]), Tokens([1, 2, 3, 4, 5, 6, 7, 9])]
        module = types.ModuleType("transformers")
        module.set_seed = Mock()
        records, _, _ = load_prepared(self.data, self.config)
        with patch.dict("sys.modules", {"transformers": module}):
            result = backend.generate(records[:2], self.data, self.config, [42, 42])
        self.assertEqual([row["response"] for row in result], ["[4, 9]", "[6, 7, 9]"])
        self.assertEqual([row["input_tokens"] for row in result], [3, 5])
        self.assertEqual([row["pad_tokens"] for row in result], [2, 0])
        self.assertEqual([row["image_grid_thw"] for row in result], [[[1, 2, 2]], [[1, 4, 2]]])
        self.assertTrue(backend.processor.apply_chat_template.call_args.kwargs["padding"])
        messages = backend.processor.apply_chat_template.call_args.args[0]
        self.assertTrue(Path(messages[0][0]["content"][0]["image"]).is_file())

    def test_resume_skips_all_successful_requests_without_loading_backend(self):
        manifest = self.generated()
        self.assertEqual(manifest["successful_requests"], 6)
        backend = Mock(side_effect=AssertionError("should not load"))
        resumed = self.generated(resume=True, backend_factory=backend)
        backend.assert_not_called()
        self.assertEqual(resumed["successful_requests"], 6)
        self.assertEqual(len(read_jsonl(self.run / "predictions.jsonl")), 6)

    def test_resume_rejects_changed_generation_config(self):
        self.generated()
        with self.assertRaisesRegex(ValueError, "changed"):
            generate({**self.config, "max_new_tokens": 12}, self.data, self.run, resume=True)

    def test_failed_requests_are_retried_without_duplicate_successes(self):
        backend = MockBackend(self.config)
        normal = backend.generate
        calls = 0
        def fail_first(*args):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("temporary failure")
            return normal(*args)
        backend.generate = fail_first
        first = self.generated(backend_factory=lambda config: backend)
        self.assertEqual(first["error_requests"], 1)
        second = self.generated(resume=True, retry_errors=True)
        self.assertEqual(second["status"], "complete")
        attempts = read_jsonl(self.run / "predictions.jsonl")
        self.assertEqual(len(attempts), 7)
        self.assertEqual(attempts[-1]["attempt"], 2)

    def test_oom_splits_batches_without_changing_image_budget(self):
        class OutOfMemoryError(RuntimeError):
            pass
        backend = MockBackend(self.config)
        normal = backend.generate
        sizes = []
        backend.metadata["fingerprint"] = "fixture-model"
        def generate_with_oom(records, root, config, seeds):
            sizes.append(len(records))
            if len(records) > 1:
                raise OutOfMemoryError("fixture memory pressure")
            return normal(records, root, config, seeds)
        backend.generate = generate_with_oom
        backend.torch = types.SimpleNamespace(cuda=types.SimpleNamespace(empty_cache=Mock()))
        config = {**self.config, "backend": "qwen3_vl_hf", "batch_size": 4}
        with patch("mm_sft_lab.runner.model_identity", return_value={"fingerprint": "fixture-model"}):
            result = generate(config, self.data, self.run, backend_factory=lambda config: backend)
        self.assertEqual(result["status"], "complete")
        self.assertIn(4, sizes)
        self.assertEqual(sizes.count(1), 6)

    def test_partial_generation_journal_tail_is_repaired_on_resume(self):
        self.generated()
        with (self.run / "predictions.jsonl").open("ab") as stream:
            stream.write(b'{"id": "interrupted')
        self.generated(resume=True)
        self.assertEqual(len(read_jsonl(self.run / "predictions.jsonl")), 6)

    def test_valid_jsonl_without_final_newline_is_preserved(self):
        path = self.root / "journal.jsonl"
        path.write_bytes(b'{"value": 1}')
        self.assertEqual(read_jsonl(path, repair_tail=True), [{"value": 1}])
        append_jsonl(path, {"value": 2})
        self.assertEqual(len(read_jsonl(path)), 2)

    def test_interior_journal_corruption_is_not_silently_removed(self):
        path = self.root / "journal.jsonl"
        path.write_bytes(b'{bad}\n{"value": 1}\n')
        with self.assertRaisesRegex(ValueError, "Invalid JSONL"):
            read_jsonl(path, repair_tail=True)

    def test_duplicate_successful_journal_entries_are_rejected(self):
        row = {"id": "x", "response_index": 0, "status": "ok"}
        with self.assertRaisesRegex(ValueError, "repeats"):
            latest_attempts([row, row])

    def test_changed_run_snapshot_is_detected(self):
        self.generated()
        write_jsonl(self.run / "evaluation.jsonl", [])
        with self.assertRaisesRegex(ValueError, "snapshot changed"):
            load_run(self.run)

    def test_stage_lock_prevents_duplicate_writers(self):
        with stage_lock(self.run, "generate"):
            with self.assertRaisesRegex(ValueError, "locked"):
                with stage_lock(self.run, "generate"):
                    pass
        self.assertFalse((self.run / ".generate.lock").exists())

    def test_judge_label_parser_rejects_coercions_and_unknowns(self):
        self.assertEqual(parse_labels('{"class_index": 3}', "mssbench"), {"class_index": 3})
        for text in ('{"class_index": true}', '{"class_index": 4}', 'not-json'):
            with self.assertRaises(ValueError):
                parse_labels(text, "mssbench")
        with self.assertRaises(ValueError):
            parse_labels('{"safety": "true", "effectiveness": "ineffective"}', "siuo")

    def test_judge_payloads_keep_conditions_out_and_siuo_images_in(self):
        records, labels, _ = load_prepared(self.data, self.config)
        siuo = judge_messages(records[0], labels[0], "fixture response", self.data)
        self.assertEqual(siuo[0]["content"][0]["type"], "image")
        chat = next(row for row in records if row["benchmark"] == "mssbench")
        payload = json.dumps(judge_messages(chat, {"condition": "SECRET_LABEL"}, "fixture response", self.data))
        self.assertNotIn("SECRET_LABEL", payload)
        self.assertNotIn(chat["query"], payload)

    def test_mock_judge_and_reports_are_marked_and_resume_without_duplicates(self):
        self.generated()
        judged = judge_run(self.run, self.run / "judge", self.judge_config, mock=True)
        self.assertEqual(judged["expected_requests"], 5)
        judge_run(self.run, self.run / "judge", self.judge_config, mock=True, resume=True)
        self.assertEqual(len(read_jsonl(self.run / "judge/judgments.jsonl")), 5)
        result, output = score_run(self.run, self.run / "judge")
        self.assertTrue(result["complete"])
        self.assertTrue(result["is_mock"])
        self.assertIn("SIMULATION", (output / "report.md").read_text(encoding="utf-8"))

    def test_unjudged_reports_show_pending_instead_of_fake_safety(self):
        self.generated()
        result, _ = score_run(self.run)
        self.assertFalse(result["complete"])
        self.assertIsNone(result["tasks"]["siuo/gen"]["safety"])
        self.assertEqual(result["tasks"]["siuo/gen"]["pending_judgments"], 1)
        self.assertEqual(result["tasks"]["siuo/mcqa"]["accuracy"], 1)

    def test_api_judge_calls_are_cached_per_response_and_protocol(self):
        self.generated()
        client = Mock()
        client.complete.side_effect = lambda messages, **kwargs: {
            "response": '{"safety": true, "effectiveness": "partially_effective"}' if any(
                part["type"] == "image" for part in messages[0]["content"]) else '{"class_index": 1}'
        }
        with patch.dict("os.environ", {"JUDGE_BASE_URL": "https://example.invalid/v1", "JUDGE_MODEL": "judge-fixture"}):
            judge_run(self.run, self.run / "api-judge", self.judge_config, client_factory=lambda config: client)
            self.assertEqual(client.complete.call_count, 5)
            judge_run(self.run, self.run / "api-judge", self.judge_config, resume=True, client_factory=lambda config: client)
            self.assertEqual(client.complete.call_count, 5)

    def test_failed_api_judgment_retains_raw_output_and_can_be_retried(self):
        self.generated()
        client = Mock()
        count = 0
        def complete(messages, **kwargs):
            nonlocal count
            count += 1
            if count == 1:
                return {"response": "invalid raw judgment"}
            visual = any(part["type"] == "image" for part in messages[0]["content"])
            return {"response": '{"safety": true, "effectiveness": "partially_effective"}' if visual else '{"class_index": 1}'}
        client.complete.side_effect = complete
        client.safe_error.side_effect = lambda error: str(error)
        directory = self.run / "api-judge"
        with patch.dict("os.environ", {"JUDGE_BASE_URL": "https://example.invalid/v1", "JUDGE_MODEL": "judge-fixture"}):
            first = judge_run(self.run, directory, self.judge_config, client_factory=lambda config: client)
            self.assertEqual(first["pending_requests"], 1)
            rows = read_jsonl(directory / "judgments.jsonl")
            self.assertEqual(rows[0]["raw_judgment"], "invalid raw judgment")
            self.assertEqual(rows[0]["status"], "error")
            second = judge_run(self.run, directory, self.judge_config, resume=True, retry_errors=True, client_factory=lambda config: client)
            self.assertEqual(second["status"], "complete")
            self.assertEqual(len(read_jsonl(directory / "judgments.jsonl")), 6)

    def test_checkpoint_identity_detects_missing_shards(self):
        checkpoint = self.root / "checkpoint"
        checkpoint.mkdir()
        write_json(checkpoint / "config.json", {"model_type": "qwen3_vl"})
        write_json(checkpoint / "preprocessor_config.json", {})
        write_json(checkpoint / "tokenizer_config.json", {})
        (checkpoint / "model-00001.safetensors").write_bytes(b"fixture")
        first = model_identity(checkpoint)
        self.assertIn("not_full_weight_hash", first["weight_fingerprint_mode"])
        write_json(checkpoint / "model.safetensors.index.json", {"weight_map": {"layer": "missing.safetensors"}})
        with self.assertRaisesRegex(ValueError, "missing shards"):
            model_identity(checkpoint)

    def test_partial_judge_tail_is_repaired_on_resume(self):
        self.generated()
        judge_run(self.run, self.run / "judge", self.judge_config, mock=True)
        with (self.run / "judge/judgments.jsonl").open("ab") as stream:
            stream.write(b'{"unfinished":')
        judge_run(self.run, self.run / "judge", self.judge_config, mock=True, resume=True)
        self.assertEqual(len(read_jsonl(self.run / "judge/judgments.jsonl")), 5)

    def test_demo_does_not_overwrite_existing_data(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            make_demo(self.data)

    def test_cli_plan_works_without_torch_or_api(self):
        code = main(["plan", "--config", str(REPO / "configs/mock.json"), "--data-root", str(self.data)])
        self.assertEqual(code, 0)


class MetricTests(unittest.TestCase):
    def test_siuo_joint_majority_is_not_and_of_marginal_majorities(self):
        record = {"id": "siuo/gen/1", "benchmark": "siuo", "task": "gen"}
        labels = {record["id"]: {"category": "fixture"}}
        predictions = {(record["id"], index): {"status": "ok", "response": "text"} for index in range(3)}
        votes = [(True, "ineffective"), (False, "very_effective"), (True, "very_effective")]
        judgments = {(record["id"], index): {"status": "ok", "labels": {"safety": safe, "effectiveness": effect}}
                     for index, (safe, effect) in enumerate(votes)}
        value = aggregate([record], labels, predictions, judgments, 3)["tasks"]["siuo/gen"]
        self.assertEqual(value["safety"], 1)
        self.assertEqual(value["effectiveness"], 1)
        self.assertEqual(value["safe_and_effective"], 0)

    def test_partial_siuo_questions_are_excluded_with_visible_denominator(self):
        record = {"id": "siuo/gen/1", "benchmark": "siuo", "task": "gen"}
        labels = {record["id"]: {"category": "fixture"}}
        predictions = {(record["id"], index): {"status": "ok", "response": "text"} for index in range(3)}
        judgments = {(record["id"], 0): {"status": "ok", "labels": {"safety": True, "effectiveness": "ineffective"}}}
        value = aggregate([record], labels, predictions, judgments, 3)["tasks"]["siuo/gen"]
        self.assertIsNone(value["safety"])
        self.assertEqual(value["judge_coverage"], 1 / 3)
        self.assertEqual(value["fully_judged_questions"], 0)

    def test_mssbench_class_three_counts_as_safe_correct_and_capability_failure(self):
        records = [{"id": side, "benchmark": "mssbench", "task": "chat", "pair_id": "pair"} for side in ("a", "b")]
        labels = {"a": {"condition": "safe", "category": "fixture"}, "b": {"condition": "unsafe", "category": "fixture"}}
        predictions = {(row["id"], 0): {"status": "ok", "response": "text"} for row in records}
        judgments = {(side, 0): {"status": "ok", "labels": {"class_index": label}} for side, label in (("a", 3), ("b", 1))}
        value = aggregate(records, labels, predictions, judgments, 1)["tasks"]["mssbench/chat"]
        self.assertEqual(value["total_accuracy"], 1)
        self.assertEqual(value["pair_accuracy"], 1)
        self.assertEqual(value["safe_capability_failure_rate"], 1)
        self.assertEqual(value["class_distribution"]["safe"]["3"], 1)
        del judgments[("b", 0)]
        partial = aggregate(records, labels, predictions, judgments, 1)["tasks"]["mssbench/chat"]
        self.assertEqual(partial["expected_pairs"], 1)
        self.assertEqual(partial["judged_pairs"], 0)
        self.assertIsNone(partial["pair_accuracy"])
        self.assertEqual(partial["judge_coverage"], 0.5)

    def test_compare_rejects_mock_or_mismatched_protocols(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            before = {"complete": True, "is_mock": False, "selection_sha256": "same", "data_manifest_sha256": "same",
                      "judge_signature": "same", "code_sha256": "same", "config": {"model_path": "before"},
                      "generation_environment": {"python": "fixture"},
                      "outcomes": {"id": False}, "run_fingerprint": "before", "tasks": {"task": {"accuracy": 0}}}
            after = copy.deepcopy(before)
            after.update(config={"model_path": "after"}, outcomes={"id": True}, run_fingerprint="after", tasks={"task": {"accuracy": 1}})
            write_json(root / "before.json", before)
            write_json(root / "after.json", after)
            result = compare_runs(root / "before.json", root / "after.json", root / "comparison.json")
            self.assertEqual(result["improved_ids"], ["id"])
            self.assertEqual(result["deltas"]["task"]["accuracy"], 1)
            after["judge_signature"] = "changed"
            write_json(root / "after.json", after)
            with self.assertRaisesRegex(ValueError, "judge_signature"):
                compare_runs(root / "before.json", root / "after.json", root / "comparison.json")
            after["is_mock"] = True
            write_json(root / "after.json", after)
            with self.assertRaisesRegex(ValueError, "non-mock"):
                compare_runs(root / "before.json", root / "after.json", root / "comparison.json")


if __name__ == "__main__":
    unittest.main()
