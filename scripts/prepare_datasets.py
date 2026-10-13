"""Validate benchmark assets and separate model inputs from evaluation labels."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.data_utils import (
    read_json, relative_media_path, sha256_file, utc_now,
    validate_source_lock, write_json_atomic, write_text_atomic,
)


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def nonempty_text(value, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value


def nonempty_list(value, label: str) -> list:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a nonempty list")
    return value


class AssetIndex:
    def __init__(self, data_root: Path):
        self.data_root = data_root.resolve()
        self.assets: dict[str, dict] = {}
        self._siuo_names = None

    def siuo_image(self, value: str) -> str:
        relative = relative_media_path(value)
        base = self.data_root / "raw" / "siuo" / "images"
        direct = base / relative
        if direct.is_file():
            return self.add(direct, base)
        if self._siuo_names is None:
            self._siuo_names = defaultdict(list)
            for path in sorted(base.rglob("*")):
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                    self._siuo_names[path.name].append(path)
        matches = self._siuo_names[relative.name]
        if len(matches) != 1:
            raise ValueError(f"SIUO image {value!r}: expected one match, found {len(matches)}")
        return self.add(matches[0], base)

    def mssbench_image(self, task: str, value: str) -> str:
        base = self.data_root / "raw" / "mssbench" / task
        return self.add(base / relative_media_path(value), base)

    def add(self, path: Path, base: Path) -> str:
        path = path.resolve()
        if not path.is_relative_to(base.resolve()):
            raise ValueError(f"Image escapes dataset directory: {path}")
        if not path.is_file():
            raise ValueError(f"Missing image: {path}")
        key = path.relative_to(self.data_root).as_posix()
        if key not in self.assets:
            try:
                from PIL import Image
                with Image.open(path) as image:
                    image.verify()
                with Image.open(path) as image:
                    image.load()
                    width, height = image.size
                    mode = image.mode
            except ImportError as error:
                raise RuntimeError("Install data dependencies: python -m pip install -r requirements-data.txt") from error
            except Exception as error:
                raise ValueError(f"Unreadable image: {path}: {error}") from error
            self.assets[key] = {
                "path": key, "sha256": sha256_file(path), "bytes": path.stat().st_size,
                "width": width, "height": height, "mode": mode,
            }
        return key


def build_records(data_root: Path) -> tuple[list, list, dict]:
    data_root = data_root.resolve()
    assets = AssetIndex(data_root)
    inputs, evaluation = [], []
    scene_counts = {}
    annotation_hashes = {}

    for task, filename in (("gen", "siuo_gen.json"), ("mcqa", "siuo_mcqa.json")):
        path = data_root / "raw" / "siuo" / "annotations" / filename
        rows = nonempty_list(read_json(path), f"SIUO {task}")
        annotation_hashes[path.relative_to(data_root).as_posix()] = sha256_file(path)
        seen = set()
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError(f"SIUO {task} row must be an object")
            qid = row.get("question_id")
            if isinstance(qid, bool) or not isinstance(qid, (str, int)) or not str(qid).strip():
                raise ValueError("SIUO question_id must be a nonempty string or integer")
            qid = str(qid)
            if qid in seen:
                raise ValueError(f"Duplicate SIUO {task} question_id: {qid}")
            seen.add(qid)
            image = assets.siuo_image(row["image"])
            record = {
                "id": f"siuo/{task}/{qid}", "benchmark": "siuo", "task": task,
                "source_scene_id": f"siuo/{qid}", "pair_id": None,
                "query": nonempty_text(row.get("question"), "SIUO question"),
                "media_paths": [image],
            }
            label = {
                "id": record["id"],
                "category": nonempty_text(row.get("category"), "SIUO category"),
            }
            if task == "gen":
                label["safety_warning"] = nonempty_text(row.get("safety_warning"), "SIUO safety_warning")
                reference = row.get("reference_answer", row.get("reference"))
                label["reference_answer"] = nonempty_text(reference, "SIUO reference_answer")
            else:
                options = row.get("options")
                if not isinstance(options, dict) or len(options) < 2:
                    raise ValueError("SIUO options must be an object with at least two choices")
                for key, value in options.items():
                    nonempty_text(key, "option label")
                    nonempty_text(value, "option text")
                if row.get("correct_option") not in options:
                    raise ValueError("SIUO correct_option is absent from options")
                record["options"] = options
                label["correct_option"] = row["correct_option"]
            inputs.append(record)
            evaluation.append(label)
        scene_counts[f"siuo_{task}"] = len(rows)

    mss_root = data_root / "raw" / "mssbench"
    available = [path for path in (mss_root / "combined.json", mss_root / "combine.json") if path.is_file()]
    if not available:
        raise ValueError(f"Missing MSSBench combined.json (or combine.json) in {mss_root}")
    if len(available) == 2 and sha256_file(available[0]) != sha256_file(available[1]):
        raise ValueError("MSSBench combined.json and combine.json differ; resolve the ambiguity")
    path = available[0]
    data = read_json(path)
    if not isinstance(data, dict):
        raise ValueError("MSSBench annotations must be an object")
    annotation_hashes[path.relative_to(data_root).as_posix()] = sha256_file(path)
    for task in ("chat", "embodied"):
        rows = nonempty_list(data.get(task), f"MSSBench {task}")
        scene_counts[f"mssbench_{task}"] = len(rows)
        for scene_index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise ValueError(f"MSSBench {task} row must be an object")
            scene_id = f"mssbench/{task}/{scene_index:04d}"
            category = row.get("Type", row.get("category"))
            if not (isinstance(category, str) and category.strip()) and not (
                isinstance(category, list) and category and all(isinstance(x, str) and x.strip() for x in category)
            ):
                raise ValueError(f"Missing or invalid category in {scene_id}")
            if task == "chat":
                safe_path, unsafe_path = row["safe_image_path"], row["unsafe_image_path"]
                queries = nonempty_list(row.get("queries"), f"{scene_id} queries")
                instructions = [(query, query) for query in queries]
            else:
                safe_path, unsafe_path = row["safe"], row["unsafe"]
                safe = nonempty_list(row.get("safe_instructions"), f"{scene_id} safe_instructions")
                unsafe = nonempty_list(row.get("unsafe_instructions"), f"{scene_id} unsafe_instructions")
                if len(safe) != len(unsafe):
                    raise ValueError(f"Unequal instruction list lengths in {scene_id}: {len(safe)} vs {len(unsafe)}")
                instructions = list(zip(safe, unsafe))
            safe_image = assets.mssbench_image(task, safe_path)
            unsafe_image = assets.mssbench_image(task, unsafe_path)
            if safe_image == unsafe_image or assets.assets[safe_image]["sha256"] == assets.assets[unsafe_image]["sha256"]:
                raise ValueError(f"Identical safe/unsafe images in {scene_id}")
            for query_index, queries in enumerate(instructions):
                pair_id = f"{scene_id}/{query_index}"
                for side, condition, query, image in (
                    ("a", "safe", queries[0], safe_image),
                    ("b", "unsafe", queries[1], unsafe_image),
                ):
                    record = {
                        "id": f"{pair_id}/{side}", "benchmark": "mssbench", "task": task,
                        "source_scene_id": scene_id, "query_index": query_index,
                        "pair_id": pair_id, "query": nonempty_text(query, f"{pair_id} instruction"),
                        "media_paths": [image],
                    }
                    inputs.append(record)
                    evaluation.append({"id": record["id"], "category": category, "condition": condition})

    ids = [record["id"] for record in inputs]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate normalized sample IDs")
    # SIUO tasks are independent; report cross-task differences without dropping rows.
    gen = {record["source_scene_id"]: record for record in inputs if record["benchmark"] == "siuo" and record["task"] == "gen"}
    mcqa = {record["source_scene_id"]: record for record in inputs if record["benchmark"] == "siuo" and record["task"] == "mcqa"}
    shared = set(gen) & set(mcqa)
    summary = {
        "scene_counts": scene_counts,
        "request_counts": dict(Counter(f"{row['benchmark']}_{row['task']}" for row in inputs)),
        "mssbench_pair_count": sum(1 for row in evaluation if row.get("condition") == "safe"),
        "unique_image_count": len(assets.assets),
        "siuo_task_overlap": {
            "shared_ids": len(shared), "gen_only_ids": sorted(set(gen) - set(mcqa)),
            "mcqa_only_ids": sorted(set(mcqa) - set(gen)),
            "different_inputs": sorted(key for key in shared if (
                gen[key]["media_paths"] != mcqa[key]["media_paths"] or gen[key]["query"] != mcqa[key]["query"]
            )),
        },
        "images": list(assets.assets.values()), "annotation_sha256": annotation_hashes,
    }
    return inputs, evaluation, summary


def check_provenance(data_root: Path, annotation_hashes: dict) -> dict:
    sources_path = data_root / "sources.lock.json"
    receipt_path = data_root / "download_manifest.json"
    if not sources_path.exists() and not receipt_path.exists():
        return {"sources": None, "provenance_status": "manual_unlocked"}
    if not sources_path.is_file() or not receipt_path.is_file():
        raise ValueError("Incomplete download provenance; rerun download_datasets.py with the same data root")
    lock = validate_source_lock(read_json(sources_path))
    receipt = read_json(receipt_path)
    if not isinstance(receipt, dict) or receipt.get("schema_version") != 1:
        raise ValueError("Invalid download manifest schema")
    if receipt.get("sources_lock_sha256") != sha256_file(sources_path) or receipt.get("sources") != lock["sources"]:
        raise ValueError("Source lock differs from the download manifest")
    expected = receipt.get("annotation_sha256", {})
    if not isinstance(expected, dict) or any(expected.get(path) != digest for path, digest in annotation_hashes.items()):
        raise ValueError("Annotations differ from the download manifest; use a new data root for modified data")
    return {
        "sources": lock, "provenance_status": "locked",
        "download_manifest_sha256": sha256_file(receipt_path),
    }


def prepare(data_root: Path, output_root: Path | None = None) -> dict:
    data_root = data_root.resolve()
    output_root = (output_root or data_root / "prepared").resolve()
    if output_root == data_root or output_root.is_relative_to(data_root / "raw"):
        raise ValueError("Prepared output must be separate from raw data")
    # Complete all validation before replacing any successful output files.
    inputs, evaluation, summary = build_records(data_root)
    provenance = check_provenance(data_root, summary["annotation_sha256"])
    texts = {
        "inputs.jsonl": "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in inputs),
        "evaluation.jsonl": "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in evaluation),
    }
    for name, text in texts.items():
        write_text_atomic(output_root / name, text)
    manifest = {
        "schema_version": 1, "prepared_at": utc_now(), "status": "validated",
        "data_root_at_preparation": str(data_root),
        "media_path_base": "data_root", "benchmark_setting": "mssbench_if",
        **provenance,
        **summary,
        "output_sha256": {name: sha256_file(output_root / name) for name in texts},
    }
    write_json_atomic(output_root / "manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    try:
        manifest = prepare(args.data_root, args.output_root)
    except Exception as error:
        print(f"Preparation failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps({key: manifest[key] for key in (
        "status", "provenance_status", "scene_counts", "request_counts", "mssbench_pair_count", "unique_image_count",
    )}, ensure_ascii=False, indent=2))
    print(f"Prepared files: {(args.output_root or args.data_root / 'prepared').resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
