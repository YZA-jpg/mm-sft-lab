"""Paired standalone scripts: no GPU, subprocess launch, real sockets or paid API calls."""

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
import urllib.error
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import start_vllm_model as launch
import smoke_vllm_benchmarks as smoke


def response(text="Synthetic answer", finish="stop"):
    return io.BytesIO(json.dumps({"model": "target", "choices": [{"message": {"content": text}, "finish_reason": finish}],
                                  "usage": {"prompt_tokens": 10, "completion_tokens": 2}}).encode())


class ScriptTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.data = self.root / "data"
        prepared = self.data / "prepared"
        prepared.mkdir(parents=True)
        picture = self.data / "image.png"
        Image.new("RGB", (32, 32), "blue").save(picture)
        rows, labels = [], []
        for index in range(2):
            identity = f"siuo/gen/{index:04d}"
            rows.append({"id": identity, "benchmark": "siuo", "task": "gen", "source_scene_id": identity,
                         "pair_id": None, "query": "Synthetic question", "media_paths": ["image.png"]})
            labels.append({"id": identity, "reference_answer": "Must never enter the target payload"})
        for task in ("chat", "embodied"):
            for index in range(2):
                pair = f"mssbench/{task}/{index:04d}/0"
                for side, condition in (("a", "safe"), ("b", "unsafe")):
                    identity = pair + "/" + side
                    rows.append({"id": identity, "benchmark": "mssbench", "task": task, "source_scene_id": pair,
                                 "pair_id": pair, "query": "Synthetic task", "media_paths": ["image.png"]})
                    labels.append({"id": identity, "condition": condition})
        for name, values in (("inputs.jsonl", rows), ("evaluation.jsonl", labels)):
            (prepared / name).write_text("".join(json.dumps(row) + "\n" for row in values))
        self.manifest = {"schema_version": 1, "status": "validated", "provenance_status": "locked", "is_mock": False,
            "output_sha256": {name: launch.sha_file(prepared / name) for name in ("inputs.jsonl", "evaluation.jsonl")},
            "images": [{"path": "image.png", "sha256": launch.sha_file(picture)}]}
        launch.write_json(prepared / "manifest.json", self.manifest)
        model = self.root / "model"
        model.mkdir()
        launch.write_json(model / "config.json", {"model_type": "qwen3_vl"})
        (model / "model.safetensors").write_bytes(b"synthetic weights, never loaded")
        settings = {"model": launch.checkpoint_identity(model)}
        self.server = self.root / "server"
        launch.write_json(self.server / "server.json", {"model_path": str(model), "model_id": "target",
            "settings": settings, "signature": launch.digest(settings), "base_url": "http://127.0.0.1:18000/v1",
            "pid": 123456, "launch_id": "synthetic-launch", "log": "synthetic.log", "start_ticks": "123", "uid": 1000})

    def tearDown(self):
        self.tmp.cleanup()

    def test_start_command_uses_fixed_cache_and_no_decoding_overrides(self):
        args = Namespace(gpu="0", port=18000, python=Path("python"), model_path=Path("model"), memory=.65, kv_cache_gib=40)
        command = launch.command(args)
        self.assertEqual(command[command.index("--kv-cache-memory-bytes") + 1], str(40 * 1024 ** 3))
        self.assertEqual(json.loads(command[command.index("--limit-mm-per-prompt") + 1]), {"image": 1, "video": 0})
        for flag in ("--max-model-len", "--generation-config", "--temperature", "--seed"):
            self.assertNotIn(flag, command)

    def test_pid_reuse_is_rejected_without_sending_a_signal(self):
        with patch.object(launch, "owned_alive", return_value=False), \
             patch.object(launch, "process_identity", return_value={"start_ticks": "different"}), \
             patch.object(launch.os, "killpg", create=True) as kill:
            with self.assertRaisesRegex(ValueError, "PID identity differs"):
                launch.stop_server(self.server)
        kill.assert_not_called()

    def test_detached_start_records_pid_and_reuses_its_existing_server(self):
        python = self.root / "venv-python"
        python.write_bytes(b"synthetic interpreter, never executed")
        args = Namespace(gpu="0", port=18000, python=python,
                         model_path=Path(launch.read_json(self.server / "server.json")["model_path"]),
                         memory=.65, kv_cache_gib=40, server_dir=self.root / "new-server")
        process = Namespace(pid=123456)
        with patch.object(launch.sys, "platform", "linux"), \
             patch.object(launch, "runtime_versions", return_value={"python": "test", "packages": {"vllm": "0.11.0"}}), \
             patch.object(launch.socket, "create_connection", side_effect=ConnectionRefusedError), \
             patch.object(launch.subprocess, "Popen", return_value=process) as spawn, \
             patch.object(launch, "process_identity", return_value={"start_ticks": "abc", "uid": 1000}), \
             patch.object(launch, "owned_alive", return_value=True):
            receipt = launch.start(args)
            reused = launch.start(args)
        self.assertEqual(receipt["launch_id"], reused["launch_id"])
        self.assertEqual(spawn.call_count, 1)
        self.assertTrue(spawn.call_args.kwargs["start_new_session"])
        self.assertEqual(spawn.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "0")
        self.assertEqual(launch.read_json(args.server_dir / "server.json")["pid"], 123456)
        self.assertEqual(receipt["generation_parameters_sent"], [])

    def test_virtualenv_executable_is_not_resolved_to_its_base_interpreter(self):
        base = self.root / "base-python"
        base.write_bytes(b"synthetic executable, never run")
        venv = self.root / "venv-python"
        try:
            venv.symlink_to(base)
        except OSError:
            self.skipTest("OS does not permit symlinks")
        try:
            args = Namespace(gpu="0", port=18000, python=venv, model_path=Path("model"), memory=.65, kv_cache_gib=40)
            self.assertEqual(launch.command(args)[0], os.path.abspath(venv))
            self.assertNotEqual(launch.command(args)[0], str(base.resolve()))
        finally:
            venv.unlink(missing_ok=True)

    def test_owned_process_requires_birth_token_uid_group_and_nonce(self):
        receipt = launch.read_json(self.server / "server.json")
        identity = {"start_ticks": "123", "uid": 1000, "pgid": 123456,
                    "environment": [b"MM_SFT_VLLM_LAUNCH_ID=synthetic-launch"]}
        with patch.object(launch, "process_identity", return_value=identity):
            self.assertTrue(launch.owned_alive(receipt))
            identity["environment"] = []
            self.assertFalse(launch.owned_alive(receipt))

    def test_limit_keeps_both_conditions_and_siuo_three_trials(self):
        jobs, _ = smoke.select_jobs(self.data, 1)
        self.assertEqual(len(jobs), 7)
        self.assertEqual(len(smoke.select_jobs(self.data, 2)[0]), 14)
        self.assertEqual(sum(j["record"]["benchmark"] == "siuo" for j in jobs), 3)

    def test_payload_preserves_image_and_never_includes_references(self):
        jobs, manifest = smoke.select_jobs(self.data, 1)
        payload = smoke.make_payload(jobs[0]["record"], self.data, manifest, "target")
        self.assertEqual(set(payload), {"model", "messages"})
        self.assertNotIn("Must never enter", json.dumps(payload))
        self.assertTrue(payload["messages"][0]["content"][0]["image_url"]["url"].startswith("data:image/png;base64,"))

    def test_network_budget_is_five_retries_and_not_reset_after_interruption(self):
        calls, events = [], []
        def opener(*args, **kwargs):
            calls.append(1)
            raise urllib.error.URLError("offline timeout")
        result = smoke.completion("http://127.0.0.1:18000/v1", {"model": "target", "messages": []}, 10,
                                  on_attempt=events.append, opener=opener, sleeper=lambda _: None)
        self.assertEqual(len(calls), 6)
        self.assertEqual([v["number"] for v in events], [1, 2, 3, 4, 5, 6])
        calls.clear()
        smoke.completion("http://127.0.0.1:18000/v1", {"model": "target", "messages": []}, 10,
                         offset=4, opener=opener, sleeper=lambda _: None)
        self.assertEqual(len(calls), 2)

    def test_provider_guardrail_is_not_retried(self):
        calls = []
        def opener(*args, **kwargs):
            calls.append(1)
            raise urllib.error.HTTPError("url", 400, "guard", {}, io.BytesIO(b'{"error":{"code":"data_inspection_failed"}}'))
        result = smoke.completion("http://127.0.0.1:18000/v1", {"model": "target", "messages": []}, 10,
                                  opener=opener, sleeper=lambda _: None)
        self.assertEqual(len(calls), 1)
        self.assertEqual(result["error_kind"], "provider_guardrail")

    def test_seven_requests_outputs_resume_and_timing_without_any_judge(self):
        run = self.root / "run"
        calls = []
        def opener(request, **kwargs):
            body = json.loads(request.data)
            self.assertEqual(set(body), {"model", "messages"})
            self.assertEqual(request.full_url, "http://127.0.0.1:18000/v1/chat/completions")
            calls.append(body)
            return response()
        args = ["--data-root", str(self.data), "--server-dir", str(self.server), "--run-dir", str(run)]
        with patch.object(smoke, "wait_for_server", return_value=0), \
             patch.object(smoke.urllib.request, "urlopen", side_effect=opener), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(smoke.main(args), 0)
            self.assertEqual(smoke.main(args + ["--resume"]), 0)
        self.assertEqual(len(calls), 7)
        summary = launch.read_json(run / "summary.json")
        self.assertEqual(summary["successful_requests"], 7)
        self.assertEqual(summary["judge_calls"], 0)
        self.assertEqual(summary["new_requests_this_session"], 0)
        self.assertEqual(len(summary["by_task"]), 3)
        self.assertTrue((run / "answers.txt").exists())

    def test_truncation_is_retained_as_needs_review(self):
        run = self.root / "truncated"
        with patch.object(smoke, "wait_for_server", return_value=0), \
             patch.object(smoke.urllib.request, "urlopen", side_effect=lambda *args, **kwargs: response(finish="length")), \
             contextlib.redirect_stdout(io.StringIO()):
            code = smoke.main(["--data-root", str(self.data), "--server-dir", str(self.server), "--run-dir", str(run)])
        self.assertEqual(code, 2)
        self.assertEqual(launch.read_json(run / "summary.json")["issue_counts"]["truncated"], 7)


if __name__ == "__main__":
    unittest.main()
