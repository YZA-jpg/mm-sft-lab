"""Target-only output budgets, frozen comparisons and four-request scheduling; no sockets."""

import contextlib
import copy
import io
import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation_v2.cli import main
from evaluation_v2.common import read_journal
from evaluation_v2.config import load, validate
from evaluation_v2.demo import FakeHTTP, make_demo
from evaluation_v2.metrics import summarize
from evaluation_v2.reporting import comparison_identity
from evaluation_v2.runner import initialize, stage
from evaluation_v2.status import status
from evaluation_v2.transport import Client


class GenerationBudgetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.data = self.root / "data"
        make_demo(self.data)
        self.endpoint = {"model": "target", "base_url": "http://offline.invalid/v1", "api_key_env": None}

    def configured(self):
        return validate({"generation": {"max_tokens": 65536}, "execution": {"generation_concurrency": 4}})

    def test_budget_validation_rejects_other_generation_controls(self):
        self.assertIsNone(validate({})["generation"]["max_tokens"])
        self.assertEqual(self.configured()["generation"]["max_tokens"], 65536)
        for value in (True, 0, -1, 1.5, "65536"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate({"generation": {"max_tokens": value}})
        for override in ({"generation": {"temperature": 0}}, {"deepseek": {"max_tokens": 65536}},
                         {"wildguard": {"max_tokens": 65536}}):
            with self.subTest(override=override), self.assertRaises(ValueError):
                validate(override)

    def test_target_only_budget_does_not_leak_to_either_judge(self):
        config, fake, run = self.configured(), FakeHTTP(), self.root / "run"
        initialize(self.data, run, config, {"kind": "mock"})
        with contextlib.redirect_stdout(io.StringIO()):
            for phase in ("generation", "deepseek", "wildguard"):
                stage(run, phase, client_factory=fake.factory)
        for call in fake.calls:
            body = call["body"]
            self.assertEqual(body.get("max_tokens"), 65536 if body["model"] == "target" else None)
            expected = {"model", "prompt" if "prompt" in body else "messages"}
            if body["model"] == "target":
                expected.add("max_tokens")
            self.assertEqual(set(body), expected)

    def test_stream_length_is_successful_preserved_and_not_retried(self):
        calls = []
        def opener(request, **kwargs):
            body = json.loads(request.data)
            calls.append(body)
            self.assertEqual(set(body), {"model", "messages", "max_tokens", "stream", "stream_options"})
            self.assertEqual(body["max_tokens"], 65536)
            event = {"choices": [{"delta": {"content": "partial but retained"}, "finish_reason": "length"}],
                     "usage": {"completion_tokens": 65536}}
            return io.BytesIO(("data: " + json.dumps(event) + "\n\ndata: [DONE]\n\n").encode())
        client = Client(self.endpoint, validate({})["execution"], opener=opener)
        result = client.complete(messages=[], max_tokens=65536, stream=True, stream_directory=self.root / "capture")
        self.assertEqual(len(calls), 1)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["finish_reason"], "length")
        self.assertEqual(result["text"], "partial but retained")
        self.assertEqual((self.root / "capture/attempt-01/response.txt").read_text(), result["text"])
        for invalid in (False, 0, -1, 1.5):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                client.complete(messages=[], max_tokens=invalid)
        self.assertEqual(len(calls), 1)

    def test_default_json_requests_still_omit_every_decoding_parameter(self):
        calls = []
        def opener(request, **kwargs):
            calls.append(json.loads(request.data))
            return io.BytesIO(b'{"choices":[{"message":{"content":"A"},"finish_reason":"stop"}]}')
        Client(self.endpoint, validate({})["execution"], opener=opener).complete(messages=[])
        self.assertEqual(set(calls[0]), {"model", "messages"})

    def test_retry_preserves_the_cap_and_remaining_transport_budget(self):
        calls = []
        def opener(request, **kwargs):
            calls.append(json.loads(request.data))
            if len(calls) == 1:
                raise TimeoutError("synthetic read timeout")
            return io.BytesIO(b'{"choices":[{"message":{"content":"A"},"finish_reason":"stop"}]}')
        client = Client(self.endpoint, validate({})["execution"], opener=opener, sleeper=lambda _: None)
        result = client.complete(messages=[], max_tokens=65536, attempt_offset=4)
        self.assertEqual(result["status"], "ok")
        self.assertEqual([row["number"] for row in result["attempts"]], [5, 6])
        self.assertEqual([row["max_tokens"] for row in calls], [65536, 65536])

    def test_resume_and_comparison_bind_the_output_budget(self):
        config, run = self.configured(), self.root / "run"
        initialize(self.data, run, config, {"kind": "mock"})
        different = copy.deepcopy(config)
        different["generation"]["max_tokens"] = 32768
        with self.assertRaisesRegex(ValueError, "Existing run differs"):
            initialize(self.data, run, different, {"kind": "mock"}, resume=True)
        first = summarize(run)
        second = copy.deepcopy(first)
        second["config"] = different
        self.assertNotEqual(comparison_identity(first), comparison_identity(second))

    def test_four_requests_overlap_with_identical_budget(self):
        config, fake, run = self.configured(), FakeHTTP(), self.root / "concurrent"
        initialize(self.data, run, config, {"kind": "mock"})
        barrier, guard = threading.Barrier(4), threading.Lock()
        active = maximum = calls = 0
        def opener(request, **kwargs):
            nonlocal active, maximum, calls
            self.assertEqual(json.loads(request.data)["max_tokens"], 65536)
            with guard:
                active += 1
                calls += 1
                first_four = calls <= 4
                maximum = max(maximum, active)
            try:
                if first_four:
                    barrier.wait(timeout=3)
                return fake(request, **kwargs)
            finally:
                with guard:
                    active -= 1
        def factory(endpoint, execution):
            return Client({**endpoint, "api_key_env": None}, execution, opener=opener)
        with contextlib.redirect_stdout(io.StringIO()):
            result = stage(run, "generation", client_factory=factory)
        self.assertTrue(result["scoring_complete"])
        self.assertEqual(maximum, 4)

    def test_truncations_remain_valid_and_visible_in_status_and_report(self):
        config, fake, run = self.configured(), FakeHTTP(), self.root / "length"
        initialize(self.data, run, config, {"kind": "mock"})
        def opener(request, **kwargs):
            value = json.load(fake(request, **kwargs))
            value["choices"][0]["finish_reason"] = "length"
            return io.BytesIO(json.dumps(value).encode())
        def factory(endpoint, execution):
            return Client({**endpoint, "api_key_env": None}, execution, opener=opener)
        with contextlib.redirect_stdout(io.StringIO()):
            generated = stage(run, "generation", client_factory=factory)
        self.assertEqual(generated["finish_reasons"], {"length": 26})
        report = summarize(run)
        self.assertEqual(report["generation"]["successful"], 26)
        self.assertEqual(report["generation"]["parameters"], {"max_tokens": 65536})
        self.assertEqual(report["generation"]["truncation"]["x"], 26)
        self.assertEqual(report["generation"]["truncation"]["y"], 26)
        self.assertEqual(report["generation"]["by_benchmark"]["mssbench"]["truncation"]["value"], 1)
        live = status(run)
        self.assertEqual(live["target_generation_parameters"], {"max_tokens": 65536})
        self.assertEqual(live["phases"]["generation"]["truncated"], 26)
        self.assertEqual(len(read_journal(run / "generation/transport-attempts.jsonl")), 26)

    def test_checked_in_configs_and_plan_expose_target_only_budget(self):
        repo = Path(__file__).resolve().parents[1]
        future = load(repo / "configs/eval_v2.64k_full.json")
        self.assertEqual(future["generation"], {"max_tokens": 65536})
        self.assertEqual(len(future["benchmarks"]), 8)
        for label, limit in (("smoke", 3), ("full", None)):
            path = repo / f"configs/eval_v2.siuo_mss_64k_{label}.json"
            config = load(path)
            self.assertEqual(config["generation"], {"max_tokens": 65536})
            self.assertEqual(config["execution"]["generation_concurrency"], 4)
            self.assertEqual(config["execution"]["per_benchmark_limit"], limit)
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                main(["plan", "--config", str(path), "--data-root", str(self.data)])
            planned = json.loads(out.getvalue())
            self.assertEqual(planned["http_decoding_parameters"], ["max_tokens"])
            self.assertEqual(planned["target_generation_parameters"], {"max_tokens": 65536})
            self.assertEqual(planned["judge_generation_parameters"], {"deepseek": {}, "wildguard": {}})


if __name__ == "__main__":
    unittest.main()
