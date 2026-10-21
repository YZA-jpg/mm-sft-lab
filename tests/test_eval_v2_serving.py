"""Managed scheduling/cleanup and concurrency sweep exercised without a real subprocess or GPU."""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation_v2.common import read_json, write_json
from evaluation_v2.config import validate
from evaluation_v2.demo import FakeHTTP, make_demo
from evaluation_v2.profiling import sweep
from evaluation_v2.runner import stage as real_stage
from evaluation_v2.serving import run_all, server_command


class ManagedTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.data = self.root / "data"
        make_demo(self.data)
        self.fake = FakeHTTP()
        self.events, self.active = [], set()
        self.identity = {"kind": "mock", "label": "SIMULATED SERVER", "config_sha256": {}}

    def tearDown(self):
        self.tmp.cleanup()

    @contextlib.contextmanager
    def serve(self, python, model, name, port, gpu, output, **kwargs):
        if self.active:
            raise AssertionError("Two GPU services loaded together")
        self.active.add(name)
        self.events.append("start:" + name)
        try:
            yield {"startup_seconds": 0}
        finally:
            self.active.remove(name)
            self.events.append("stop:" + name)

    def stage(self, run, phase, **kwargs):
        if phase == "generation":
            self.assertEqual(self.active, {"target"})
        if phase == "wildguard":
            self.assertEqual(self.active, {"wildguard"})
        self.events.append("phase:" + phase)
        return real_stage(run, phase, client_factory=self.fake.factory, **kwargs)

    def test_one_card_schedule_stops_target_before_wildguard_and_resume_loads_no_model(self):
        config = validate({})
        with patch("evaluation_v2.serving.checkpoint_identity", side_effect=lambda *args, **kwargs: self.identity.copy()), \
             patch("evaluation_v2.serving.runtime_info", return_value={"packages": {"vllm": "test"}}), \
             patch("evaluation_v2.serving.serve", side_effect=self.serve), \
             patch("evaluation_v2.serving.Monitor", side_effect=lambda *args: contextlib.nullcontext()), \
             patch("evaluation_v2.serving.stage", side_effect=self.stage), contextlib.redirect_stdout(io.StringIO()):
            result = run_all(self.data, self.root / "run", config, self.root / "model", self.root / "guard")
            self.assertTrue(result["complete"])
            self.assertLess(self.events.index("stop:target"), self.events.index("start:wildguard"))
            count = len(self.fake.calls)
            events = len(self.events)
            run_all(self.data, self.root / "run", config, self.root / "model", self.root / "guard", resume=True)
            self.assertEqual(len(self.fake.calls), count)
            self.assertFalse(any(e.startswith("start:") for e in self.events[events:]))

    def test_owned_service_context_closes_on_generation_failure(self):
        def failure(*args, **kwargs):
            raise RuntimeError("simulated inference failure")
        with patch("evaluation_v2.serving.checkpoint_identity", side_effect=lambda *args, **kwargs: self.identity.copy()), \
             patch("evaluation_v2.serving.runtime_info", return_value={}), \
             patch("evaluation_v2.serving.serve", side_effect=self.serve), \
             patch("evaluation_v2.serving.Monitor", side_effect=lambda *args: contextlib.nullcontext()), \
             patch("evaluation_v2.serving.stage", side_effect=failure):
            with self.assertRaisesRegex(RuntimeError, "inference failure"):
                run_all(self.data, self.root / "failed", validate({}), self.root / "model", self.root / "guard")
        self.assertFalse(self.active)
        self.assertEqual(self.events[-1], "stop:target")

    def test_sweep_same_inputs_every_level_no_cloud_judge_or_cached_generation(self):
        with patch("evaluation_v2.profiling.checkpoint_identity", return_value=self.identity.copy()), \
             patch("evaluation_v2.profiling.runtime_info", return_value={}), \
             patch("evaluation_v2.profiling.serve", side_effect=self.serve), \
             patch("evaluation_v2.profiling.Monitor", side_effect=lambda *args: contextlib.nullcontext()), \
             patch("evaluation_v2.profiling.stage", side_effect=self.stage), contextlib.redirect_stdout(io.StringIO()):
            rows = sweep(self.data, self.root / "profile", validate({}), self.root / "model", levels=(1, 2), rounds=1, limit=1)
        plans = [read_json(path) for path in (self.root / "profile").glob("round-*/manifest.json")]
        self.assertEqual(len(plans), 2)
        self.assertEqual(len({p["signature"]["selection_sha256"] for p in plans}), 1)
        self.assertTrue(all(call["body"]["model"] == "target" for call in self.fake.calls))
        self.assertEqual([event for event in self.events if event.startswith("start:")], ["start:target"])
        self.assertTrue((self.root / "profile/sweep.csv").exists())
        self.assertEqual(len([r for r in rows if r["benchmark"] == "ALL"]), 2)

    def test_generate_only_has_no_judge_calls_and_resumes_into_full_scoring(self):
        with patch("evaluation_v2.serving.checkpoint_identity", side_effect=lambda *args, **kwargs: self.identity.copy()), \
             patch("evaluation_v2.serving.runtime_info", return_value={}), \
             patch("evaluation_v2.serving.serve", side_effect=self.serve), \
             patch("evaluation_v2.serving.Monitor", side_effect=lambda *args: contextlib.nullcontext()), \
             patch("evaluation_v2.serving.stage", side_effect=self.stage), contextlib.redirect_stdout(io.StringIO()):
            args = (self.data, self.root / "generation-only", validate({}), self.root / "model", self.root / "guard")
            result = run_all(*args, target_kv_cache_gib=40, generate_only=True)
            self.assertEqual(result["generation"]["successful"], 26)
            self.assertFalse(result["complete"])
            self.assertTrue(all(c["body"]["model"] == "target" for c in self.fake.calls))
            self.assertEqual(self.events, ["start:target", "phase:generation", "stop:target"])
            self.assertEqual(result["model_identity"]["serving_runtime"]["target_kv_cache_gib"], 40)
            result = run_all(*args, target_kv_cache_gib=40, resume=True)
            self.assertTrue(result["complete"])
            self.assertEqual(len(self.fake.calls), 50)
            with self.assertRaisesRegex(ValueError, "Existing run differs"):
                run_all(*args, target_kv_cache_gib=32, resume=True)

    def test_fixed_cache_is_an_operational_option_not_a_decoding_override(self):
        args = server_command("python", "/model", "target", 18000, memory=.65, kv_cache_gib=40, image_only=True)
        self.assertEqual(args[args.index("--kv-cache-memory-bytes") + 1], str(40 * 1024 ** 3))
        self.assertEqual(json.loads(args[args.index("--limit-mm-per-prompt") + 1]), {"image": 1, "video": 0})
        for flag in ("--max-model-len", "--generation-config", "--seed", "--temperature"):
            self.assertNotIn(flag, args)
        guard = server_command("python", "/guard", "wildguard", 18001, kv_cache_gib=8)
        self.assertNotIn("--limit-mm-per-prompt", guard)
        for value in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                server_command("python", "/model", "target", 18000, kv_cache_gib=value)

    def test_cli_generate_only_does_not_require_cloud_credentials(self):
        from evaluation_v2.cli import main
        report = {"generation": {"successful": 7, "expected": 7}, "complete": False}
        with patch("evaluation_v2.cli.configuration.load", return_value=validate({"benchmarks": ["siuo", "mssbench"]})), \
             patch("evaluation_v2.cli.run_all", return_value=report) as run, \
             patch.dict("os.environ", {"DASHSCOPE_API_KEY": ""}), contextlib.redirect_stdout(io.StringIO()):
            code = main(["run", "--data-root", "data", "--run-dir", "run", "--model-path", "model",
                         "--generate-only", "--target-kv-cache-gib", "40"])
        self.assertEqual(code, 0)
        self.assertTrue(run.call_args.kwargs["generate_only"])
        self.assertEqual(run.call_args.kwargs["target_kv_cache_gib"], 40)


if __name__ == "__main__":
    unittest.main()
