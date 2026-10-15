"""Sequential source failures must not prevent independent downloads."""

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
from scripts.download_step1 import main
from mm_sft_lab.common import read_json


class BatchDownloadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for stream in ("stdout", "stderr"):
            quiet = getattr(contextlib, f"redirect_{stream}")(io.StringIO())
            quiet.__enter__()
            self.addCleanup(quiet.__exit__, None, None, None)

    def test_first_failure_does_not_skip_other_sources_and_returns_failure_after_all(self):
        calls = []
        def download(root, name, **kwargs):
            calls.append(name)
            if name == "textvqa":
                raise TimeoutError("fixture network failure")
            return {"source": name, "files": {"one": {}}}
        with patch("scripts.download_step1.download_source", side_effect=download):
            code = main(["--all", "--data-root", str(self.root)])
        self.assertEqual(calls, ["textvqa", "hh", "wildjailbreak", "vlguard"])
        self.assertEqual(code, 1)
        result = read_json(self.root / "download_batch_summary.json")
        self.assertEqual(result["status"], "partial_failure")
        self.assertEqual([row["status"] for row in result["datasets"]], ["failed", "downloaded", "downloaded", "downloaded"])
        self.assertIn("TimeoutError", result["datasets"][0]["error"])

    def test_wild_permission_failure_preserves_other_successes(self):
        def download(root, name, **kwargs):
            if name == "wildjailbreak":
                raise PermissionError("fixture dataset access not granted")
            return {"source": name, "files": {}}
        with patch("scripts.download_step1.download_source", side_effect=download):
            self.assertEqual(main(["--all", "--data-root", str(self.root)]), 1)
        result = read_json(self.root / "download_batch_summary.json")
        self.assertEqual([row["status"] for row in result["datasets"]], ["downloaded", "downloaded", "failed", "downloaded"])

    def test_all_successes_return_zero_and_forward_worker_count(self):
        with patch("scripts.download_step1.download_source", return_value={"files": {}}) as downloader:
            self.assertEqual(main(["--all", "--data-root", str(self.root), "--max-workers", "2"]), 0)
        self.assertEqual(downloader.call_count, 4)
        self.assertTrue(all(call.kwargs["max_workers"] == 2 for call in downloader.call_args_list))
        self.assertEqual(read_json(self.root / "download_batch_summary.json")["status"], "complete")

    def test_all_failures_still_attempt_every_source(self):
        with patch("scripts.download_step1.download_source", side_effect=OSError("fixture error")) as downloader:
            self.assertEqual(main(["--all", "--data-root", str(self.root)]), 1)
        self.assertEqual(downloader.call_count, 4)

    def test_vlguard_access_failure_preserves_the_three_completed_sources(self):
        def download(root, name, **kwargs):
            if name == "vlguard":
                raise PermissionError("fixture VLGuard access not granted")
            return {"source": name, "files": {}}
        with patch("scripts.download_step1.download_source", side_effect=download):
            self.assertEqual(main(["--all", "--data-root", str(self.root)]), 1)
        result = read_json(self.root / "download_batch_summary.json")
        self.assertEqual([row["status"] for row in result["datasets"]], ["downloaded", "downloaded", "downloaded", "failed"])

    def test_single_vlguard_cli_is_available(self):
        with patch("scripts.download_step1.download_source", return_value={"source": "vlguard", "files": {}}) as downloader:
            self.assertEqual(main(["--dataset", "vlguard", "--data-root", str(self.root)]), 0)
        self.assertEqual(downloader.call_args.args[1], "vlguard")

    def test_single_source_cli_is_preserved(self):
        with patch("scripts.download_step1.download_source", return_value={"source": "hh", "files": {}}) as downloader:
            self.assertEqual(main(["--dataset", "hh", "--data-root", str(self.root), "--revision", "main"]), 0)
        self.assertEqual(downloader.call_args.args[1], "hh")
        self.assertEqual(downloader.call_args.kwargs["revision"], "main")

    def test_batch_rejects_revision_override_before_starting_downloads(self):
        with patch("scripts.download_step1.download_source") as downloader:
            with self.assertRaises(SystemExit) as error:
                main(["--all", "--revision", "main"])
        self.assertEqual(error.exception.code, 2)
        downloader.assert_not_called()

    def test_user_interrupt_stops_the_batch_and_records_partial_progress(self):
        with patch("scripts.download_step1.download_source", side_effect=KeyboardInterrupt) as downloader:
            self.assertEqual(main(["--all", "--data-root", str(self.root)]), 130)
        self.assertEqual(downloader.call_count, 1)
        self.assertEqual(read_json(self.root / "download_batch_summary.json")["status"], "interrupted")


if __name__ == "__main__":
    unittest.main()
