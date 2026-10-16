"""Processor compatibility contracts without torch, model files or network."""

import contextlib
import io
import json
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from mm_sft_lab.training_data import load_processor, load_train_config, processor_check


class ProbeImageProcessor:
    """Expose synthetic grids, without performing or claiming real Qwen resizing."""
    patch_size = 16
    merge_size = 2
    size = {"shortest_edge": 65536, "longest_edge": 1048576}
    min_pixels = 65536
    max_pixels = 1048576

    def __init__(self, grids=None):
        self.grids = grids
        self.input_sizes = []

    def __call__(self, *, images, return_tensors):
        self.input_sizes.append(images.size)
        edge = 16 if images.width < 256 else 64
        return {"image_grid_thw": [[1, edge, edge]] if self.grids is None else self.grids}


class TrainingProcessorTests(unittest.TestCase):
    def setUp(self):
        self.config = load_train_config(REPO / "configs/train/textvqa_lora.json")

    def make_processor(self, *, do_resize=True):
        image = types.SimpleNamespace(size={"shortest_edge": 4096, "longest_edge": 16777216},
                                      min_pixels=4096, max_pixels=16777216,
                                      do_resize=do_resize, patch_size=16, merge_size=2)
        return types.SimpleNamespace(image_processor=image, tokenizer=types.SimpleNamespace(padding_side="left"))

    def test_load_synchronizes_size_with_legacy_pixel_fields(self):
        processor = self.make_processor()

        def from_pretrained(*args, **kwargs):
            # Transformers 4.57.1 can update legacy attributes after construction,
            # leaving the size dictionary at its checkpoint values.
            for key in ("min_pixels", "max_pixels"):
                if key in kwargs:
                    setattr(processor.image_processor, key, kwargs[key])
            return processor

        loader = Mock(side_effect=from_pretrained)
        with patch.dict(sys.modules, {"transformers": types.SimpleNamespace(AutoProcessor=types.SimpleNamespace(from_pretrained=loader))}):
            result = load_processor(self.config)
        self.assertIs(result, processor)
        self.assertEqual(result.image_processor.size, {"shortest_edge": 65536, "longest_edge": 1048576})
        self.assertEqual(result.image_processor.min_pixels, 65536)
        self.assertEqual(result.image_processor.max_pixels, 1048576)
        self.assertEqual(result.tokenizer.padding_side, "right")
        self.assertTrue(loader.call_args.kwargs["local_files_only"])
        self.assertFalse(loader.call_args.kwargs["trust_remote_code"])

    def test_load_rejects_disabled_resize_even_when_size_matches(self):
        processor = self.make_processor(do_resize=False)
        processor.image_processor.size = {"shortest_edge": 65536, "longest_edge": 1048576}
        loader = Mock(return_value=processor)
        with patch.dict(sys.modules, {"transformers": types.SimpleNamespace(AutoProcessor=types.SimpleNamespace(from_pretrained=loader))}):
            with self.assertRaisesRegex(ValueError, "resize"):
                load_processor(self.config)

    def test_load_also_supports_checkpoint_without_legacy_pixel_attributes(self):
        processor = self.make_processor()
        del processor.image_processor.min_pixels
        del processor.image_processor.max_pixels
        loader = Mock(return_value=processor)
        with patch.dict(sys.modules, {"transformers": types.SimpleNamespace(AutoProcessor=types.SimpleNamespace(from_pretrained=loader))}):
            result = load_processor(self.config)
        self.assertEqual(result.image_processor.min_pixels, 65536)
        self.assertEqual(result.image_processor.max_pixels, 1048576)

    def test_processor_check_observes_both_actual_grid_boundaries(self):
        image = ProbeImageProcessor()
        result = processor_check(self.config, processor=types.SimpleNamespace(image_processor=image))
        self.assertEqual(result["status"], "passed")
        self.assertEqual(image.input_sizes, [(64, 64), (2048, 2048)])
        self.assertEqual([probe["resized_pixels"] for probe in result["probes"]], [65536, 1048576])
        self.assertEqual([probe["visual_tokens"] for probe in result["probes"]], [64, 1024])

    def test_processor_check_rejects_actual_grid_outside_budget_despite_matching_config(self):
        for grid in ([[1, 2, 2]], [[1, 128, 128]]):
            with self.subTest(grid=grid):
                processor = types.SimpleNamespace(image_processor=ProbeImageProcessor(grid))
                with self.assertRaisesRegex(ValueError, "outside training pixel budget"):
                    processor_check(self.config, processor=processor)

    def test_processor_check_rejects_missing_or_invalid_image_grid(self):
        for grid in ([], [[2, 16, 16]], [[1, 17, 16]], [[1, -1, 16]]):
            with self.subTest(grid=grid):
                processor = types.SimpleNamespace(image_processor=ProbeImageProcessor(grid))
                with self.assertRaisesRegex(ValueError, "grid"):
                    processor_check(self.config, processor=processor)

    def test_processor_check_cli_runs_without_training_dependencies_or_data_root(self):
        from scripts import train
        output = io.StringIO()
        with patch.object(sys, "argv", ["train.py", "processor-check", "--config", str(REPO / "configs/train/textvqa_lora.json")]), \
                patch.object(train, "processor_check", return_value={"status": "passed"}) as check, \
                patch.dict(sys.modules, {"mm_sft_lab.training": None}), contextlib.redirect_stdout(output):
            self.assertEqual(train.main(), 0)
        check.assert_called_once_with(self.config)
        self.assertEqual(json.loads(output.getvalue()), {"status": "passed"})


if __name__ == "__main__":
    unittest.main()
