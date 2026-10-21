"""Native source schemas, immutable assets, safe extraction and service command planning."""

import csv
import io
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation_v2.common import asset_path, file_hash, read_journal, write_json
from evaluation_v2.config import validate
from evaluation_v2.data import (FIGSTEP_RULES, Writer, import_figstep, import_holisafe, import_mathvista,
                                import_mmstar, import_mmvet, literal_strings)
from evaluation_v2.serving import checkpoint_identity, server_command
from evaluation_v2.sources import download_all, safe_extract, verify_sources


class NativeDataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.image = self.root / "source.png"
        Image.new("RGB", (32, 32), "green").save(self.image)
        self.writer = Writer(self.root / "prepared")

    def tearDown(self):
        self.tmp.cleanup()

    def test_literal_source_is_read_without_execution(self):
        source = self.root / "upstream.py"
        source.write_text("PROMPT = 'literal'\nraise RuntimeError('MUST NOT EXECUTE')\nDYNAMIC = str(3)\n")
        self.assertEqual(literal_strings(source), {"PROMPT": "literal"})
        source = self.root / "source.ipynb"
        write_json(source, {"cells": [{"cell_type": "code", "source": ["prompt = 'notebook'\n", "raise RuntimeError()"]}]})
        self.assertEqual(literal_strings(source, notebook=True)["prompt"], "notebook")

    def test_images_keep_original_bytes_and_are_deduplicated(self):
        first = self.writer.image("source.png", self.root)
        second = self.writer.image(self.image.read_bytes(), self.root)
        self.assertEqual(first, second)
        self.assertEqual((self.writer.root / first).read_bytes(), self.image.read_bytes())
        self.assertEqual(len(self.writer.assets), 1)

    def test_asset_paths_and_zip_cannot_escape_output(self):
        for name in ("../escape", "/absolute", "D:/absolute", "..\\escape"):
            with self.assertRaises(ValueError):
                asset_path(self.root, name)
        archive = self.root / "bad.zip"
        with zipfile.ZipFile(archive, "w") as stream:
            stream.writestr("../escape.txt", "bad")
        with self.assertRaises(ValueError):
            safe_extract(archive, self.root / "extracted")

    def test_figstep_rule_mapping_and_exact_original_image_variant(self):
        folder = self.root / "data/images/SafeBench"
        folder.mkdir(parents=True)
        (folder / "query_ForbidQI_6_1_6.png").write_bytes(self.image.read_bytes())
        path = self.root / "data/question/safebench.csv"
        path.parent.mkdir()
        path.write_text("dataset,category_id,task_id,category_name,question,instruction\nForbidQI,6,1,Adult Content,Synthetic question,Synthetic instruction\n")
        source = self.root / "rules.py"
        source.write_text("Sex_rule_safety = 'Synthetic rule'\n")
        row = import_figstep(self.writer, self.root, source)[0]
        self.assertEqual(row["reference"]["scenario_rule"], "Synthetic rule")
        self.assertEqual(row["reference"]["image_variant"], 6)
        path.write_text(path.read_text().replace("Adult Content", "Fraud"))
        with self.assertRaisesRegex(ValueError, "mismatch"):
            import_figstep(self.writer, self.root, source)

    def test_holisafe_five_groups(self):
        write_json(self.root / "holisafe_bench.json", [{"id": i, "type": kind, "query": "Synthetic query", "image": "source.png"}
                   for i, kind in enumerate(("SSS", "SUU", "USU", "UUU", "SSU"))])
        rows = import_holisafe(self.writer, self.root)
        self.assertEqual({r["split"] for r in rows}, {"SSS", "SUU", "USU", "UUU", "SSU"})

    def test_mmstar_parquet_image_bytes(self):
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            self.skipTest("Optional pyarrow data dependency unavailable")
        pq.write_table(pa.Table.from_pylist([{"index": 0, "question": "Synthetic? A. One B. Two", "answer": "A",
                         "image": {"bytes": self.image.read_bytes(), "path": None}, "category": "synthetic"}]), self.root / "data.parquet")
        rows = import_mmstar(self.writer, self.root)
        self.assertEqual(rows[0]["reference"]["answer"], "A")
        self.assertEqual(len(rows[0]["images"]), 1)

    def test_mathvista_requires_testmini_and_reads_answer_type(self):
        (self.root / "testmini.parquet").write_bytes(b"fixture")
        values = [{"pid": "1", "image": "source.png", "question": "Two plus two?", "query": "Two plus two?",
                   "answer": "4", "question_type": "free_form", "answer_type": "integer", "precision": 0, "choices": None}]
        with patch("evaluation_v2.data.parquet_rows", return_value=values):
            rows = import_mathvista(self.writer, self.root)
        self.assertEqual(rows[0]["split"], "testmini")
        self.assertEqual(rows[0]["reference"]["answer_type"], "integer")

    def test_mmvet_v1_schema(self):
        write_json(self.root / "mm-vet.json", {"v1_0": {"imagename": "source.png", "question": "Synthetic?", "answer": "green", "capability": ["recognition"]}})
        rows = import_mmvet(self.writer, self.root)
        self.assertEqual(rows[0]["id"], "mmvet/v1/v1_0")

    def test_download_failure_does_not_block_other_sources(self):
        def download(name, root):
            if name == "holisafe":
                raise PermissionError("Gated access needed")
            return {"status": "downloaded", "files": 1}
        with patch("evaluation_v2.sources.download_source", side_effect=download), patch("builtins.print"):
            result = download_all(["holisafe", "mmstar", "mathvista"], self.root)
        self.assertEqual(result["holisafe"]["status"], "failed")
        self.assertEqual(result["mathvista"]["status"], "downloaded")

    def test_raw_inventory_detects_mutation(self):
        folder = self.root / "native"
        folder.mkdir()
        (folder / "source.json").write_text("[]")
        write_json(folder / "download_manifest.json", {"lock": {"revision": "pinned"}, "files": {"source.json": {"sha256": file_hash(folder / "source.json")}}})
        self.assertIn("native", verify_sources(self.root))
        (folder / "source.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "source changed"):
            verify_sources(self.root)

    def test_model_identity_adapter_and_no_decoding_cli_options(self):
        model, adapter = self.root / "model", self.root / "adapter"
        model.mkdir()
        adapter.mkdir()
        write_json(model / "config.json", {"model_type": "qwen3_vl"})
        write_json(model / "generation_config.json", {"from_model": True})
        (model / "model.safetensors").write_bytes(b"synthetic weights, never loaded")
        write_json(adapter / "adapter_config.json", {"r": 16})
        (adapter / "adapter_model.safetensors").write_bytes(b"synthetic adapter")
        identity = checkpoint_identity(model, adapter=adapter)
        self.assertIn("generation_config.json", identity["config_sha256"])
        self.assertIn("adapter_model.safetensors", identity["adapter"]["files_sha256"])
        args = server_command("python", model, "target", 8000, adapter=adapter)
        self.assertIn("target_adapter=" + str(adapter), args)
        for forbidden in ("--max-model-len", "--generation-config", "--override-generation-config", "--seed", "--temperature"):
            self.assertNotIn(forbidden, args)

    def test_unknown_hyperparameters_are_rejected(self):
        for value in ({"max_tokens": 50}, {"target": {"temperature": 0}}, {"execution": {"network_retries": 4}}):
            with self.assertRaises(ValueError):
                validate(value)


if __name__ == "__main__":
    unittest.main()
