"""Create a tiny synthetic dataset for an offline pipeline demonstration."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image
from scripts.data_utils import write_json_atomic
from scripts.prepare_datasets import prepare


def make_demo(root):
    root = Path(root).resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError("Demo data root must be empty; choose a new directory")
    for relative, color in (
        ("raw/siuo/images/S-01.png", "white"),
        ("raw/mssbench/chat/safe.png", "green"), ("raw/mssbench/chat/unsafe.png", "red"),
        ("raw/mssbench/embodied/safe.png", "blue"), ("raw/mssbench/embodied/unsafe.png", "yellow"),
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (16, 16), color).save(path)
    shared = {"question_id": 1001, "image": "S-01.png", "question": "Describe the color.", "category": "demo-only"}
    write_json_atomic(root / "raw/siuo/annotations/siuo_gen.json", [
        {**shared, "safety_warning": "Synthetic warning for an offline test.", "reference_answer": "The square is white."}])
    write_json_atomic(root / "raw/siuo/annotations/siuo_mcqa.json", [
        {**shared, "options": {"A": "White", "B": "Black"}, "correct_option": "A"}])
    write_json_atomic(root / "raw/mssbench/combined.json", {
        "chat": [{"safe_image_path": "safe.png", "unsafe_image_path": "unsafe.png", "queries": ["Describe the square."], "Type": "demo-only"}],
        "embodied": [{"safe": "safe.png", "unsafe": "unsafe.png", "safe_instructions": ["Describe the blue square."],
                      "unsafe_instructions": ["Describe the yellow square."], "category": "demo-only"}],
    })
    return prepare(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/demo"))
    args = parser.parse_args()
    try:
        make_demo(args.data_root)
    except Exception as error:
        print(f"Demo preparation failed: {error}", file=sys.stderr)
        return 1
    print(f"Synthetic demo prepared: {args.data_root.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
