"""Launcher acceptance uses synthetic evidence only; never starts a process, GPU or API."""

import contextlib
import copy
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation_v2.common import append, read_journal
from evaluation_v2.config import validate
from evaluation_v2.data import Writer
from evaluation_v2.demo import FakeHTTP
from evaluation_v2.runner import initialize, stage
from scripts.start_fixed_baseline import accept_smoke, launch_detached, main


class FixedLauncherTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.data, self.run = self.root / "data", self.root / "run"
        writer = Writer(self.data)
        rows = []
        for index in range(5):
            out = io.BytesIO()
            Image.new("RGB", (32, 32), (index * 40, 20, 10)).save(out, format="WEBP")
            image = writer.image(out.getvalue(), self.root)
            rows.append({"id": f"mssbench/chat/{index}/a", "benchmark": "mssbench", "split": "chat",
                         "query": "Synthetic mssbench question", "images": [image],
                         "reference": {"pair_id": str(index), "condition": "safe"}})
            rows.append({**rows[-1], "id": f"mssbench/chat/{index}/b",
                         "reference": {"pair_id": str(index), "condition": "unsafe"}})
        rows.append({"id": "siuo/gen/a", "benchmark": "siuo", "split": "gen", "query": "Synthetic siuo query",
                     "images": [image], "reference": {}})
        # Simulate production markers to exercise acceptance; files and outputs are synthetic temporary fixtures.
        writer.finish(rows, is_mock=False, provenance={"fixture": "synthetic acceptance regression"})
        self.full = validate({"benchmarks": ["siuo", "mssbench"], "generation": {"max_tokens": 65536},
                              "execution": {"benchmark_schedule": "serial"}})
        self.smoke = copy.deepcopy(self.full)
        self.smoke["execution"]["sample_ids"] = [r["id"] for r in rows]
        self.identity = {"kind": "local_checkpoint", "label": "SYNTHETIC FIXTURE"}
        initialize(self.data, self.run, self.smoke, self.identity)
        with contextlib.redirect_stdout(io.StringIO()):
            stage(self.run, "generation", client_factory=FakeHTTP().factory)

    def accept(self, **overrides):
        values = {"run_dir": self.run, "data_root": self.data, "full_config": self.full,
                  "smoke_config": self.smoke, "checkpoint": self.identity, **overrides}
        return accept_smoke(**values)

    def test_accepts_complete_matching_smoke_without_judging(self):
        result = self.accept()
        self.assertEqual(result["status"], "accepted_generation_smoke_only")
        self.assertEqual(result["webp_images_checked"], 5)
        self.assertEqual(result["successful_jobs"], 13)
        self.assertFalse((self.run / "deepseek/results.jsonl").exists())

    def test_wrong_model_config_or_active_writer_cannot_authorize_full(self):
        with self.assertRaisesRegex(ValueError, "Model/adapter differs"):
            self.accept(checkpoint={"kind": "local_checkpoint", "label": "different"})
        different = copy.deepcopy(self.full)
        different["execution"]["generation_concurrency"] = 8
        with self.assertRaisesRegex(ValueError, "Full configuration differs"):
            self.accept(full_config=different)
        (self.run / ".pipeline.lock").write_text("{}")
        with self.assertRaisesRegex(ValueError, "still locked"):
            self.accept()

    def test_errors_and_changed_output_cannot_authorize_full(self):
        row = dict(read_journal(self.run / "generation/results.jsonl")[0])
        row.update(attempt=2, text="tampered")
        append(self.run / "generation/results.jsonl", row)
        with self.assertRaisesRegex(ValueError, "Invalid smoke generation evidence"):
            self.accept()
        row.update(attempt=3, status="error")
        append(self.run / "generation/results.jsonl", row)
        with self.assertRaisesRegex(ValueError, "incomplete or has errors"):
            self.accept()

    def test_check_only_never_checks_gpu_launches_server_or_requires_model(self):
        with patch("scripts.start_fixed_baseline.load", return_value=self.full), \
             patch("scripts.start_fixed_baseline.subprocess.run", side_effect=AssertionError("No process")), \
             patch("scripts.start_fixed_baseline.subprocess.Popen", side_effect=AssertionError("No server")), \
             patch("scripts.start_fixed_baseline.checkpoint_identity", side_effect=AssertionError("No weights")), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["full", "--check-only", "--data-root", str(self.data)]), 0)

    def test_background_process_has_new_session_file_log_and_no_terminal_input(self):
        with patch("scripts.start_fixed_baseline.subprocess.Popen") as create:
            launch_detached(["bash", "scripts/run_eval_v2.sh"], self.root)
        self.assertTrue(create.call_args.kwargs["start_new_session"])
        self.assertEqual(create.call_args.kwargs["stdin"], -3)  # subprocess.DEVNULL
        self.assertEqual(create.call_args.kwargs["stderr"], -2)  # subprocess.STDOUT
        self.assertEqual(Path(create.call_args.kwargs["stdout"].name), self.root / "job.log")


if __name__ == "__main__":
    unittest.main()
