"""Tiny VLGuard fixture for an existing synthetic Step 1 parent, never real data."""

import copy
import io
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from PIL import Image
from mm_sft_lab.common import digest, file_hash, read_json, read_jsonl, write_json
from mm_sft_lab.vlguard import prepare_vlguard


def make_vlguard_demo(parent):
    parent = Path(parent).resolve()
    if not read_json(parent / "prepared/manifest.json").get("is_mock"):
        raise ValueError("VLGuard fixtures require a synthetic parent; never add fake data to a real root")
    raw = parent / "raw/vlguard"
    raw.mkdir(parents=True)
    def png(color):
        stream = io.BytesIO()
        Image.new("RGB", (16, 16), color).save(stream, format="PNG")
        return stream.getvalue()
    evaluation = read_jsonl(parent / "prepared/inputs.jsonl")
    eval_image = next(path for row in evaluation for path in row["media_paths"])
    unsafe = {"id": "unsafe", "image": "bad/unsafe.png", "safe": False,
              "instr-resp": [{"instruction": "Synthetic unsafe-image query.", "response": "Synthetic safe target."}]}
    rows = [
        {"id": "safe", "image": "good/safe.png", "safe": True, "instr-resp": [
            {"safe_instruction": "Describe this synthetic marker.", "response": "A synthetic marker."},
            {"unsafe_instruction": "Synthetic unsafe instruction.", "response": "Synthetic refusal."}]},
        unsafe,
        {**copy.deepcopy(unsafe), "id": "duplicate"},
        {**copy.deepcopy(unsafe), "id": "heldout-image", "image": "bad/heldout.jpg"},
        {"id": "heldout-query", "image": "bad/query.png", "safe": False, "instr-resp": [
            {"instruction": "Explain this fictional test marker.", "response": "Synthetic target."}]},
    ]
    write_json(raw / "train.json", rows)
    with zipfile.ZipFile(raw / "train.zip", "w") as bundle:
        bundle.writestr("train/good/safe.png", png((101, 20, 35)))
        bundle.writestr("train/bad/unsafe.png", png((201, 30, 45)))
        bundle.writestr("train/bad/query.png", png((77, 88, 99)))
        bundle.writestr("train/bad/heldout.jpg", (parent / eval_image).read_bytes())
    lock = {"source": "vlguard", "kind": "synthetic_fixture"}
    write_json(raw / "source.lock.json", lock)
    write_json(raw / "download_manifest.json", {"schema_version": 1, "source": "vlguard", "provenance": "manual",
        "source_lock_sha256": digest(lock), "files": {name: {"sha256": file_hash(raw / name), "bytes": (raw / name).stat().st_size}
                                                      for name in ("train.json", "train.zip")}})
    return prepare_vlguard(parent, parent / "vlguard", parent, allow_unlocked=True)
