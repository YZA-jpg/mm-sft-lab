"""Independent VLGuard question/response training snapshot, without changing eval."""

from collections import Counter
from pathlib import Path

from .common import digest, file_hash, now, read_json, stage_lock, write_json, write_jsonl
from .config import TASKS
from .data import load_prepared
from .step1_data import Images, _training, normalized_text
from .step1_sources import extract_images, inside, verify_source

TARGET_POLICY = "independent_instruction_response_last_assistant_v1"


def vlguard_rows(rows):
    if not isinstance(rows, list) or not rows:
        raise ValueError("VLGuard train.json must be a nonempty list")
    for source_index, row in enumerate(rows):
        identifier = row.get("id") if isinstance(row, dict) else None
        if not isinstance(identifier, str) or not identifier.strip() or any(c in identifier for c in "/\\:"):
            raise ValueError("VLGuard has an invalid source ID")
        safe, pairs = row.get("safe"), row.get("instr-resp")
        if type(safe) is not bool or not isinstance(pairs, list) or len(pairs) != (2 if safe else 1):
            raise ValueError(f"Unexpected VLGuard pair layout: {identifier}")
        expected = {"safe_instruction", "unsafe_instruction"} if safe else {"instruction"}
        found = set()
        for pair in pairs:
            if not isinstance(pair, dict) or "response" not in pair or len(pair) != 2:
                raise ValueError(f"Invalid VLGuard instruction/response: {identifier}")
            key = next(key for key in pair if key != "response")
            if key not in expected or key in found:
                raise ValueError(f"Unexpected/duplicate VLGuard instruction kind: {identifier}")
            found.add(key)
            query, response = pair[key], pair["response"]
            if any(not isinstance(text, str) or not text.strip() for text in (query, response)):
                raise ValueError(f"Empty VLGuard instruction/response: {identifier}/{key}")
            if any(token in text for text in (query, response) for token in
                   ("<image>", "<|im_start|>", "<|im_end|>", "<|image_pad|>", "<|vision_start|>", "<|vision_end|>")):
                raise ValueError(f"VLGuard contains model control tokens: {identifier}/{key}")
            kind = {"instruction": "unsafe_image", "safe_instruction": "safe_image_safe_instruction",
                    "unsafe_instruction": "safe_image_unsafe_instruction"}[key]
            yield {"id": f"train/vlguard/{source_index:06d}/{identifier}/{key}", "source_id": identifier, "source_index": source_index,
                   "image": row.get("image"), "image_safe": safe, "instruction_kind": kind,
                   "harmful_category": row.get("harmful_category"), "query": query.strip(), "response": response.strip()}


def prepare_vlguard(source_root, output_root, benchmark_root, *, allow_unlocked=False):
    source_root, output_root, benchmark_root = map(lambda path: Path(path).resolve(),
                                                   (source_root, output_root, benchmark_root))
    if output_root == source_root or output_root == benchmark_root or source_root.is_relative_to(output_root):
        raise ValueError("VLGuard output must be separate from existing data snapshots")
    if any(output_root.is_relative_to(root / area) for root in (source_root, benchmark_root)
           for area in ("raw", "prepared")):
        raise ValueError("VLGuard output cannot be inside existing raw/prepared directories")
    with stage_lock(output_root, "prepare_vlguard"):
        source = verify_source(source_root, "vlguard", require_locked=not allow_unlocked)
        lock = read_json(source_root / "raw/vlguard/source.lock.json")
        config = {"tasks": list(TASKS), "per_task_limit": None, "require_locked_data": not allow_unlocked}
        evaluation, _, eval_manifest = load_prepared(benchmark_root, config)
        previous_path = output_root / "prepared/manifest.json"
        if previous_path.exists():
            previous = read_json(previous_path)
            if (previous.get("source_manifest_digests") != {"vlguard": digest(source)} or
                    previous.get("audit", {}).get("evaluation_manifest_sha256") != file_hash(benchmark_root / "prepared/manifest.json")):
                raise ValueError("Prepared VLGuard sources/evaluation would change; use a new output root")
            for name, expected in previous["output_sha256"].items():
                if file_hash(inside(output_root / "prepared", name)) != expected:
                    raise ValueError(f"Prepared VLGuard snapshot changed: {name}")
        eval_media = {path for row in evaluation for path in row["media_paths"]}
        eval_hashes = {item["sha256"] for item in eval_manifest["images"] if item["path"] in eval_media}
        eval_queries = {normalized_text(row["query"]) for row in evaluation}
        rows = read_json(source_root / "raw/vlguard/train.json")
        expanded = list(vlguard_rows(rows))
        if not allow_unlocked and (len(rows), len(expanded)) != (2000, 2977):
            raise ValueError("Unexpected official VLGuard train size; inspect source before changing the protocol")
        destination = output_root / "raw/vlguard/images"
        print("[vlguard] Checking/extracting train.zip and validating training images", flush=True)
        extract_images(source_root / "raw/vlguard/train.zip", destination)
        images, training, excluded, seen = Images(output_root), [], [], set()
        for row in expanded:
            path = images.add(inside(destination / "train", row["image"]))
            key = (images.assets[path]["sha256"], normalized_text(row["query"]))
            reason = ("evaluation_image_hash" if key[0] in eval_hashes else
                      "evaluation_query_text" if key[1] in eval_queries else
                      "duplicate_training_pair" if key in seen else None)
            if reason:
                excluded.append({"id": row["id"], "source_id": row["source_id"], "reason": reason,
                                 "instruction_kind": row["instruction_kind"], "source_index": row["source_index"]})
                continue
            seen.add(key)
            record = _training(row["id"], "vlguard", [path], [
                {"role": "user", "content": row["query"]}, {"role": "assistant", "content": row["response"]}], row["source_id"])
            record.update(instruction_kind=row["instruction_kind"], image_safe=row["image_safe"],
                          harmful_category=row["harmful_category"], source_index=row["source_index"])
            training.append(record)
        if not training:
            raise ValueError("VLGuard training set is empty after filtering")
        audit = {"schema_version": 1, "status": "complete", "method": "exact_image_hash_query_within_input_v1",
                 "raw_image_records": len(rows), "unique_source_ids": len({row["id"] for row in rows}),
                 "raw_training_counts": {"vlguard": len(expanded)},
                 "training_counts": {"vlguard": len(training)}, "excluded_counts": dict(Counter(row["reason"] for row in excluded)),
                 "raw_kind_counts": dict(Counter(row["instruction_kind"] for row in expanded)),
                 "training_kind_counts": dict(Counter(row["instruction_kind"] for row in training)),
                 "unique_training_images": len({row["media_paths"][0] for row in training}),
                 "evaluation_manifest_sha256": file_hash(benchmark_root / "prepared/manifest.json"),
                 "evaluation_query_count": len(eval_queries), "evaluation_image_hash_count": len(eval_hashes),
                 "vlguard_test_overlap": "not_checked_test_not_downloaded",
                 "near_duplicate_detection": "not_implemented"}
        outputs = {"train_vlguard.jsonl": training, "excluded_training.jsonl": excluded}
        content = {"source_manifest_digests": {"vlguard": digest(source)}, "source_lock": lock,
                   "audit": audit, "training_files": {"vlguard": "train_vlguard.jsonl"},
                   "target_policy": {"vlguard": TARGET_POLICY},
                   "images": sorted(images.assets.values(), key=lambda item: item["path"]),
                   "record_digests": {name: digest(records) for name, records in outputs.items()}}
        prepared = output_root / "prepared"
        if (prepared / "manifest.json").exists():
            previous = read_json(prepared / "manifest.json")
            if any(previous.get(key) != value for key, value in content.items()):
                raise ValueError("Prepared VLGuard snapshot would change; use a new output root")
            for name, expected in previous["output_sha256"].items():
                if file_hash(inside(prepared, name)) != expected:
                    raise ValueError(f"Prepared VLGuard snapshot changed: {name}")
            return previous
        for name, records in outputs.items():
            write_jsonl(prepared / name, records)
        write_json(prepared / "decontamination.json", audit)
        manifest = {**content, "schema_version": 1, "step1_schema_version": 1, "created_at": now(), "status": "validated",
                    "is_mock": allow_unlocked, "provenance_status": "manual" if allow_unlocked else "locked",
                    "output_sha256": {name: file_hash(prepared / name) for name in (*outputs, "decontamination.json")}}
        write_json(prepared / "manifest.json", manifest)
        return manifest
