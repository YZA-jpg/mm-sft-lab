"""Tiny synthetic Step 1 sources for offline testing, not training data."""

import argparse
import csv
import gzip
import io
import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from PIL import Image
from scripts.make_demo import make_demo
from mm_sft_lab.common import digest, file_hash, write_json
from mm_sft_lab.step1_data import prepare_step1


def _jpeg(color):
    stream = io.BytesIO()
    Image.new("RGB", (16, 16), color).save(stream, format="JPEG")
    return stream.getvalue()


def make_step1_demo(root):
    root = Path(root).resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError("Step 1 demo root must be empty")
    legacy = root / "legacy"
    make_demo(legacy)
    raw = root / "raw/textvqa"
    raw.mkdir(parents=True)
    def example(qid, image, split, question):
        return {"question_id": qid, "image_id": image, "set_name": split, "question": question,
                "answers": ["Label"] * 10 if split != "test" else [""] * 10}
    write_json(raw / "TextVQA_0.5.1_train.json", {"data": [example(1, "train-1", "train", "Read the sample word."),
        example(2, "train-1", "train", "Read the sample word."), example(3, "val-1", "train", "A held-out image question."),
        example(4, "train-2", "train", "Another held-out image question.")]})
    write_json(raw / "TextVQA_0.5.1_val.json", {"data": [example(10, "val-1", "val", "Read this held-out label.")]})
    write_json(raw / "TextVQA_0.5.1_test.json", {"data": [example(20, "test-1", "test", "Read this test label.")]})
    heldout = _jpeg((32, 54, 76))
    with zipfile.ZipFile(raw / "train_val_images.zip", "w") as archive:
        archive.writestr("train_images/train-1.jpg", _jpeg((11, 22, 33)))
        archive.writestr("train_images/train-2.jpg", heldout)
        archive.writestr("train_images/val-1.jpg", heldout)
    with zipfile.ZipFile(raw / "test_images.zip", "w") as archive:
        archive.writestr("test_images/test-1.jpg", _jpeg((87, 65, 43)))
    wild = root / "raw/wildjailbreak/eval/eval.tsv"
    wild.parent.mkdir(parents=True)
    with wild.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["adversarial", "data_type"], delimiter="\t")
        writer.writeheader()
        writer.writerows([{"adversarial": "Synthetic adversarial prompt; do not act on it.", "data_type": "adversarial_harmful"},
                         {"adversarial": "Explain this fictional test marker.", "data_type": "adversarial_benign"}])
    hh = root / "raw/hh/harmless-base"
    hh.mkdir(parents=True)
    def pair(question, answer="Hello.", history=""):
        prefix = history + f"\n\nHuman: {question}\n\nAssistant: "
        return {"chosen": prefix + answer, "rejected": prefix + "An alternative synthetic answer."}
    rows = [pair("Tell me a neutral greeting."), pair("Tell me a neutral greeting."), pair("What is the held-out greeting?"),
            pair("Explain this fictional test marker."), pair("Answer with a friendly close.", "Goodbye.",
            "\n\nHuman: Start a synthetic conversation.\n\nAssistant: Earlier synthetic model output.")]
    for split, items in (("train", rows), ("test", [pair("What is the held-out greeting?")])):
        with gzip.open(hh / f"{split}.jsonl.gz", "wt", encoding="utf-8") as stream:
            for row in items:
                stream.write(json.dumps(row) + "\n")
    for name in ("textvqa", "hh", "wildjailbreak"):
        base = root / "raw" / name
        lock = {"source": name, "kind": "synthetic_fixture"}
        write_json(base / "source.lock.json", lock)
        paths = [path for path in base.rglob("*") if path.is_file() and path.name != "source.lock.json"]
        write_json(base / "download_manifest.json", {"schema_version": 1, "source": name, "provenance": "manual",
                   "source_lock_sha256": digest(lock), "files": {path.relative_to(base).as_posix():
                   {"sha256": file_hash(path), "bytes": path.stat().st_size} for path in paths}})
    return prepare_step1(root, legacy, allow_unlocked=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/step1-demo"))
    args = parser.parse_args()
    try:
        result = make_step1_demo(args.data_root)
    except Exception as error:
        print(f"Demo failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"is_mock": result["is_mock"], "training": result["audit"]["training_counts"],
                      "excluded": result["audit"]["excluded_counts"], "evaluation": result["request_counts"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
