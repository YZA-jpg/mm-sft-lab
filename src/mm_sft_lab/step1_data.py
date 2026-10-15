"""Prepare portable training/evaluation data, with exact-overlap audit trails."""

import csv
import gzip
import json
import os
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from .common import digest, file_hash, now, read_json, read_jsonl, stage_lock, write_json, write_jsonl
from .step1_sources import PREPARATION_SOURCE_NAMES, extract_images, inside, verify_source
from .textvqa import training_answer

TEXTVQA_PROMPT = "Answer the question using a short word or phrase.\nQuestion: "
TRAIN_FILES = {"textvqa": "train_textvqa.jsonl", "hh": "train_hh.jsonl"}


def normalized_text(text):
    return " ".join(text.casefold().split())


def parse_hh(transcript):
    if not isinstance(transcript, str):
        raise ValueError("HH transcripts must be strings")
    text = "\n\n" + transcript.lstrip("\n")
    matches = list(re.finditer(r"\n\n(Human|Assistant):", text))
    if not matches or text[:matches[0].start()].strip():
        raise ValueError("HH transcript does not start with Human:")
    messages = []
    for index, match in enumerate(matches):
        role = "user" if match.group(1) == "Human" else "assistant"
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        content = text[match.end():end].strip()
        if role != ("user" if index % 2 == 0 else "assistant") or not content:
            raise ValueError("HH roles must alternate and each turn must be nonempty")
        if any(token in content for token in ("<|im_start|>", "<|im_end|>", "<|image_pad|>")):
            raise ValueError("HH target contains model control tokens")
        messages.append({"role": role, "content": content})
    if not messages or messages[-1]["role"] != "assistant":
        raise ValueError("HH transcript must end in an assistant response")
    return messages


def hh_rows(path, *, excluded=None):
    """Keep strict parsing unless the caller explicitly collects rejected rows."""
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            if not line.strip():
                continue
            reason = "hh_invalid_conversation"
            try:
                value = json.loads(line)
                chosen, rejected = parse_hh(value["chosen"]), parse_hh(value["rejected"])
            except (ValueError, KeyError, TypeError) as error:
                if excluded is None:
                    raise
                detail = str(error)
            else:
                if chosen[:-1] == rejected[:-1]:
                    yield index, chosen
                    continue
                reason, detail = "hh_history_mismatch", "HH chosen/rejected history differs"
                if excluded is None:
                    raise ValueError(f"{detail} at row {index}")
            excluded.append({"source_index": index, "line_number": index + 1,
                             "reason": reason, "detail": detail})


def hh_test_contexts(path, *, issues):
    """Use both held-out branches' history even when the final target is empty."""
    contexts, raw_count = set(), 0
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            if not line.strip():
                continue
            raw_count += 1
            value = json.loads(line)
            for branch in ("chosen", "rejected"):
                try:
                    transcript = value[branch]
                    if not isinstance(transcript, str):
                        raise ValueError("HH transcripts must be strings")
                    text = "\n\n" + transcript.lstrip("\n")
                    position = text.rfind("\n\nAssistant:")
                    if position < 0:
                        raise ValueError("HH test has no final Assistant delimiter")
                    history = parse_hh(text[:position] + "\n\nAssistant: context-only placeholder")[:-1]
                except (ValueError, KeyError, TypeError) as error:
                    issues.append({"source_index": index, "line_number": index + 1, "branch": branch,
                                   "reason": "hh_test_context_unparseable", "detail": str(error)})
                else:
                    contexts.add(digest([{**turn, "content": normalized_text(turn["content"])} for turn in history]))
    return contexts, raw_count


def wild_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if not reader.fieldnames or "data_type" not in reader.fieldnames:
            raise ValueError("WildJailbreak eval TSV requires data_type")
        for index, value in enumerate(reader):
            kind = value["data_type"]
            if kind not in ("adversarial_harmful", "adversarial_benign"):
                raise ValueError(f"Unexpected WildJailbreak eval data_type: {kind}")
            query = value.get("adversarial") or value.get("prompt")
            if not isinstance(query, str) or not query.strip():
                raise ValueError("WildJailbreak eval has no adversarial prompt")
            yield index, kind.removeprefix("adversarial_"), query


class Images:
    def __init__(self, root, *, progress=False):
        self.root = Path(root).resolve()
        self.assets = {}
        self.progress = progress

    def add(self, path):
        path = Path(path).resolve()
        if not path.is_relative_to(self.root) or not path.is_file():
            raise ValueError(f"Missing/outside image: {path}")
        name = path.relative_to(self.root).as_posix()
        if name not in self.assets:
            from PIL import Image
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                image.load()
                size = image.size
            self.assets[name] = {"path": name, "sha256": file_hash(path), "bytes": path.stat().st_size,
                                 "width": size[0], "height": size[1]}
            if self.progress and len(self.assets) % 2000 == 0:
                print(f"[prepare] Validated {len(self.assets)} unique images", flush=True)
        return name


def _input(identifier, benchmark, task, query, media, scene):
    return {"id": identifier, "benchmark": benchmark, "task": task, "source_scene_id": str(scene),
            "query_index": 0, "pair_id": None, "query": query, "media_paths": media, "options": {}}


def _training(identifier, dataset, media, messages, source_id):
    return {"id": identifier, "dataset": dataset, "source_id": str(source_id), "media_paths": media,
            "messages": messages, "supervision": "last_assistant"}


def _copy_baseline(root, baseline, images, *, require_locked):
    from .data import load_prepared
    config = {
        "tasks": ["siuo/gen", "siuo/mcqa", "mssbench/chat", "mssbench/embodied"],
        "per_task_limit": None, "require_locked_data": require_locked}
    records, labels, manifest = load_prepared(baseline, config)
    hashes = {item["path"]: item["sha256"] for item in manifest["images"]}
    converted = []
    for record in records:
        paths = []
        for name in record["media_paths"]:
            target_name = "raw/baseline/" + name
            target = inside(root, target_name)
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.link(inside(baseline, name), target)
                except OSError:
                    shutil.copyfile(inside(baseline, name), target)
            media = images.add(target)
            if images.assets[media]["sha256"] != hashes[name]:
                raise ValueError("Baseline image changed while importing")
            paths.append(media)
        converted.append({**record, "media_paths": paths})
    return converted, labels, manifest


def prepare_step1(root, benchmark_root, *, allow_unlocked=False):
    root, benchmark_root = Path(root).resolve(), Path(benchmark_root).resolve()
    if root == benchmark_root:
        raise ValueError("Step 1 and original baseline must use different data roots")
    def progress(message):
        if not allow_unlocked:
            print(f"[prepare] {message}", flush=True)
    with stage_lock(root, "prepare_step1"):
        source_manifests = {}
        for name in PREPARATION_SOURCE_NAMES:
            progress(f"Checking source files and SHA256: {name}")
            source_manifests[name] = verify_source(root, name, require_locked=not allow_unlocked)
        for archive in ("train_val_images.zip", "test_images.zip"):
            progress(f"Checking/extracting {archive}")
            extract_images(root / "raw/textvqa" / archive, root / "raw/textvqa/images")
        images = Images(root, progress=not allow_unlocked)
        progress("Importing SIUO/MSSBench and validating evaluation images")
        inputs, labels, legacy = _copy_baseline(root, benchmark_root, images, require_locked=not allow_unlocked)
        eval_queries = {normalized_text(row["query"]) for row in inputs}
        eval_image_hashes = {item["sha256"] for item in images.assets.values()}
        textvqa = {}
        for split in ("train", "val", "test"):
            path = root / "raw/textvqa" / f"TextVQA_0.5.1_{split}.json"
            textvqa[split] = read_json(path)["data"]
            expected = {"train": 34602, "val": 5000, "test": 5734}[split]
            if not allow_unlocked and len(textvqa[split]) != expected:
                raise ValueError(f"Unexpected TextVQA {split} count: {len(textvqa[split])}, expected {expected}")
            if len({str(row['question_id']) for row in textvqa[split]}) != len(textvqa[split]):
                raise ValueError(f"Duplicate TextVQA {split} question_id")
        heldout_image_ids = {str(row["image_id"]) for split in ("val", "test") for row in textvqa[split]}
        for split in ("val", "test"):
            progress(f"Preparing TextVQA {split}")
            for row in textvqa[split]:
                media = images.add(root / "raw/textvqa/images" / ("test_images" if split == "test" else "train_images") / f"{row['image_id']}.jpg")
                qid = f"textvqa/{split}/{row['question_id']}"
                inputs.append(_input(qid, "textvqa", split, row["question"], [media], row["image_id"]))
                answers = row.get("answers")
                if split == "val" and (not isinstance(answers, list) or len(answers) != 10 or not any(answers)):
                    raise ValueError("TextVQA Val is missing references")
                labels.append({"id": qid, "category": "ocr", "condition": None,
                               "answers": answers if split == "val" else None, "question_id": row["question_id"]})
                eval_queries.add(normalized_text(row["question"]))
                eval_image_hashes.add(images.assets[media]["sha256"])
        progress("Preparing WildJailbreak eval")
        wild = list(wild_rows(root / "raw/wildjailbreak/eval/eval.tsv"))
        counts = Counter(kind for _, kind, _ in wild)
        if not allow_unlocked and counts != {"harmful": 2000, "benign": 210}:
            raise ValueError(f"Unexpected WildJailbreak eval counts: {dict(counts)}")
        for index, kind, query in wild:
            qid = f"wildjailbreak/{kind}/{index:05d}"
            inputs.append(_input(qid, "wildjailbreak", kind, query, [], index))
            labels.append({"id": qid, "category": "adversarial", "condition": kind})
            eval_queries.add(normalized_text(query))
        excluded, training = [], {"textvqa": [], "hh": []}
        seen_textvqa = set()
        progress("Preparing TextVQA train and checking held-out image overlap")
        for row in textvqa["train"]:
            media = images.add(root / "raw/textvqa/images/train_images" / f"{row['image_id']}.jpg")
            qid = f"train/textvqa/{row['question_id']}"
            key = (images.assets[media]["sha256"], normalized_text(row["question"]))
            reason = ("heldout_image_id" if str(row["image_id"]) in heldout_image_ids else
                      "evaluation_image_hash" if key[0] in eval_image_hashes else
                      "duplicate_training_pair" if key in seen_textvqa else None)
            if reason:
                excluded.append({"id": qid, "reason": reason})
                continue
            seen_textvqa.add(key)
            try:
                target = training_answer(row["answers"])
            except ValueError as error:
                raise ValueError(f"{qid}: {error}") from error
            training["textvqa"].append(_training(qid, "textvqa", [media], [
                {"role": "user", "content": TEXTVQA_PROMPT + row["question"]},
                {"role": "assistant", "content": target}], row["question_id"]))
        hh_base = root / "raw/hh/harmless-base"
        progress("Preparing HH chosen, auditing conversation format and held-out contexts")
        hh_test_issues, hh_format_excluded = [], []
        heldout_hh, hh_test_raw_count = hh_test_contexts(hh_base / "test.jsonl.gz", issues=hh_test_issues)
        seen_hh, hh_valid_count = set(), 0
        for index, messages in hh_rows(hh_base / "train.jsonl.gz", excluded=hh_format_excluded):
            hh_valid_count += 1
            qid = f"train/hh/{index:06d}"
            history = [{**turn, "content": normalized_text(turn["content"])} for turn in messages[:-1]]
            identity = digest([{**turn, "content": normalized_text(turn["content"])} for turn in messages])
            reason = ("hh_test_context" if digest(history) in heldout_hh else
                      "evaluation_query_text" if normalized_text(messages[-2]["content"]) in eval_queries else
                      "duplicate_chosen_conversation" if identity in seen_hh else None)
            if reason:
                excluded.append({"id": qid, "reason": reason})
                continue
            seen_hh.add(identity)
            training["hh"].append(_training(qid, "hh", [], messages, index))
        excluded.extend({"id": f"train/hh/{row['source_index']:06d}", **row} for row in hh_format_excluded)
        hh_raw_count = hh_valid_count + len(hh_format_excluded)
        if not all(training.values()):
            raise ValueError("At least one training set is empty after filtering")
        if len({row["id"] for row in inputs}) != len(inputs):
            raise ValueError("Duplicate evaluation IDs")
        audit = {"schema_version": 1, "status": "complete", "method": "exact_text_image_hash_source_id_v1",
                 "near_duplicate_detection": "not_implemented", "checks": ["TextVQA heldout image IDs", "all evaluation image hashes",
                 "HH heldout contexts from both branches", "HH strict conversation and matching histories",
                 "HH final user versus evaluation queries", "within-training exact duplicates"],
                 "raw_training_counts": {"textvqa": len(textvqa["train"]), "hh": hh_raw_count},
                 "hh_format_valid_count": hh_valid_count,
                 "hh_format_excluded_count": len(hh_format_excluded),
                 "hh_test_raw_count": hh_test_raw_count, "hh_test_context_count": len(heldout_hh),
                 "hh_test_context_issue_count": len(hh_test_issues),
                 "hh_test_context_issue_unit": "branch",
                 "training_counts": {name: len(rows) for name, rows in training.items()},
                 "excluded_counts": dict(Counter(row["reason"] for row in excluded)),
                 "baseline_manifest_sha256": file_hash(benchmark_root / "prepared/manifest.json")}
        outputs = {"inputs.jsonl": inputs, "evaluation.jsonl": labels, "excluded_training.jsonl": excluded,
                   "hh_test_context_issues.jsonl": hh_test_issues,
                   **{TRAIN_FILES[name]: rows for name, rows in training.items()}}
        prepared = root / "prepared"
        progress("Checking/writing prepared snapshot and audit files")
        previous_path = prepared / "manifest.json"
        sources = {name: digest(value) for name, value in source_manifests.items()}
        if previous_path.exists():
            previous = read_json(previous_path)
            if (previous.get("record_digests") != {name: digest(rows) for name, rows in outputs.items()} or
                    previous.get("source_manifest_digests") != sources or previous.get("audit") != audit or
                    previous.get("images") != sorted(images.assets.values(), key=lambda item: item["path"])):
                raise ValueError("Prepared snapshot would change; use a new data root")
            for name, expected in previous["output_sha256"].items():
                if file_hash(prepared / name) != expected:
                    raise ValueError(f"Prepared snapshot changed: {name}")
            return previous
        for name, rows in outputs.items():
            write_jsonl(prepared / name, rows)
        write_json(prepared / "decontamination.json", audit)
        manifest = {"schema_version": 1, "created_at": now(), "status": "validated", "step1_schema_version": 1,
                    "provenance_status": "manual" if allow_unlocked else "locked", "is_mock": allow_unlocked,
                    "source_manifest_digests": sources, "audit": audit, "training_files": TRAIN_FILES,
                    "target_policy": {"textvqa": "normalized_majority_nonblank_raw_then_frequency_lexical_tie_v2", "hh": "last_chosen_assistant_audited_strict_pairs_v2"},
                    "output_sha256": {name: file_hash(prepared / name) for name in (*outputs, "decontamination.json")},
                    "record_digests": {name: digest(rows) for name, rows in outputs.items()},
                    "images": sorted(images.assets.values(), key=lambda item: item["path"]),
                    "request_counts": dict(Counter(f"{row['benchmark']}/{row['task']}" for row in inputs))}
        write_json(previous_path, manifest)
        return manifest
