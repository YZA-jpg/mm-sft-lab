"""Read-only prepared-snapshot audit; output is separate from frozen data."""

import gzip
import json
from collections import Counter
from pathlib import Path

from .common import digest, file_hash, now, read_json, read_jsonl
from .config import TASKS
from .data import load_prepared
from .step1_data import normalized_text
from .step1_sources import inside, verify_source
from .training_data import load_train_config, load_training_records


def raw_hh_context(transcript):
    if not isinstance(transcript, str):
        raise ValueError("HH transcript must be text")
    text = "\n\n" + transcript.lstrip("\n")
    position = text.rfind("\n\nAssistant:")
    if position < 0:
        raise ValueError("HH transcript has no Assistant delimiter")
    return normalized_text(text[:position])


def verify_outputs(root, manifest):
    for name, expected in manifest["output_sha256"].items():
        if file_hash(inside(Path(root) / "prepared", name)) != expected:
            raise ValueError(f"Prepared output changed: {name}")


def audit_data(root, config_dir, *, vlguard_root=None):
    root, config_dir = Path(root).resolve(), Path(config_dir).resolve()
    manifest = read_json(root / "prepared/manifest.json")
    original_hash = file_hash(root / "prepared/manifest.json")
    verify_outputs(root, manifest)
    for name, expected in manifest["source_manifest_digests"].items():
        if digest(verify_source(root, name, require_locked=not manifest.get("is_mock"))) != expected:
            raise ValueError(f"Source manifest changed: {name}")
    evaluation, labels, _ = load_prepared(root, {"tasks": list(TASKS), "per_task_limit": None,
                                               "require_locked_data": not manifest.get("is_mock")})
    image_map = {item["path"]: item["sha256"] for item in manifest["images"]}
    eval_images = {image_map[path] for row in evaluation for path in row["media_paths"]}
    eval_queries = {normalized_text(row["query"]) for row in evaluation}
    notes, violations, groups, hh_kept = [], [], {}, set()
    roots = [("textvqa", root), ("hh", root)]
    if vlguard_root is not None:
        roots.append(("vlguard", Path(vlguard_root).resolve()))
    for dataset, training_root in roots:
        config = load_train_config(config_dir / f"{dataset}_lora.json")
        config["require_locked_data"] = not manifest.get("is_mock")
        records, training_manifest = load_training_records(training_root, config)
        verify_outputs(training_root, training_manifest)
        if dataset == "vlguard":
            if training_manifest["audit"]["evaluation_manifest_sha256"] != original_hash:
                raise ValueError("VLGuard was checked against a different evaluation snapshot")
            source = verify_source(root, "vlguard", require_locked=not manifest.get("is_mock"))
            if digest(source) != training_manifest["source_manifest_digests"]["vlguard"]:
                raise ValueError("VLGuard source manifest changed")
        images = {item["path"]: item["sha256"] for item in training_manifest["images"]}
        excluded = [row for row in read_jsonl(training_root / "prepared/excluded_training.jsonl")
                    if row["id"].startswith(f"train/{dataset}/")]
        raw = training_manifest["audit"]["raw_training_counts"][dataset]
        if raw != len(records) + len(excluded):
            violations.append(f"{dataset}: raw/retained/excluded count mismatch")
        if len({row["id"] for row in records + excluded}) != raw:
            violations.append(f"{dataset}: retained/excluded IDs overlap or are missing")
        duplicates, seen, image_overlap, query_overlap = [], set(), [], []
        for row in records:
            media_hashes = tuple(images[path] for path in row["media_paths"])
            query = normalized_text(row["messages"][-2]["content"])
            identity = digest([{**turn, "content": normalized_text(turn["content"])} for turn in row["messages"]]) if dataset == "hh" else digest([media_hashes, query])
            if identity in seen:
                duplicates.append(row["id"])
            seen.add(identity)
            if set(media_hashes) & eval_images:
                image_overlap.append(row["id"])
            if dataset in ("hh", "vlguard") and query in eval_queries:
                query_overlap.append(row["id"])
            if dataset == "hh":
                hh_kept.add(int(row["source_id"]))
        if duplicates or image_overlap or query_overlap:
            violations.append(f"{dataset}: duplicate/evaluation overlap remains")
        groups[dataset] = {"raw_samples": raw, "training_samples": len(records), "excluded_samples": len(excluded),
                           "excluded_counts": dict(Counter(row["reason"] for row in excluded)),
                           "unique_training_images": len({images[path] for row in records for path in row["media_paths"]}),
                           "duplicate_ids": duplicates[:20], "evaluation_image_overlap_ids": image_overlap[:20],
                           "evaluation_query_overlap_ids": query_overlap[:20], "optimizer_steps_at_batch32": (len(records) + 31) // 32,
                           "data_manifest_sha256": file_hash(training_root / "prepared/manifest.json")}
    issues = read_jsonl(root / "prepared/hh_test_context_issues.jsonl")
    wanted = {(row["source_index"], row["branch"]) for row in issues}
    unresolved_contexts = {}
    with gzip.open(root / "raw/hh/harmless-base/test.jsonl.gz", "rt", encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            if not line.strip():
                continue
            value = json.loads(line)
            for branch in ("chosen", "rejected"):
                if (index, branch) in wanted:
                    unresolved_contexts[(index, branch)] = raw_hh_context(value[branch])
    overlap = []
    lookup = {}
    for identity, context in unresolved_contexts.items():
        lookup.setdefault(context, []).append(identity)
    with gzip.open(root / "raw/hh/harmless-base/train.jsonl.gz", "rt", encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            if index in hh_kept:
                for test_index, branch in lookup.get(raw_hh_context(json.loads(line)["chosen"]), []):
                    overlap.append({"training_source_index": index, "test_source_index": test_index, "branch": branch})
    if overlap:
        violations.append("HH unparsed test branches share exact raw contexts with retained training rows")
    if issues:
        notes.append("HH test has unparsed branches; raw-prefix comparison supplements strict parsing but is not semantic decontamination")
    if vlguard_root is None:
        notes.append("VLGuard prepared snapshot was not supplied; its preparation audit is still pending")
    notes.extend(["Near-duplicate/semantic decontamination is not implemented",
                  "Actual Qwen token lengths, GPU updates, checkpoint resume equivalence and adapter reload remain server checks"])
    if file_hash(root / "prepared/manifest.json") != original_hash:
        raise ValueError("Evaluation snapshot changed during read-only audit")
    return {"schema_version": 1, "created_at": now(), "status": "failed" if violations else "passed_with_notes",
            "is_mock": manifest.get("is_mock", False), "data_manifest_sha256": original_hash,
            "evaluation_request_counts": dict(Counter(f"{row['benchmark']}/{row['task']}" for row in evaluation)),
            "evaluation_label_count": len(labels), "training": groups,
            "hh_unparsed_test": {"branches": len(issues), "source_rows": len({row["source_index"] for row in issues}),
                                 "exact_raw_context_overlaps": overlap}, "violations": violations, "notes": notes,
            "scope": "Read-only data integrity and exact-overlap audit; no model/API/training execution"}
