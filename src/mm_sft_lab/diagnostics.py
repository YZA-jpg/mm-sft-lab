"""Offline checks of generation journals; no model loading or API judging."""

import csv
import io
import math
from collections import Counter, defaultdict
from pathlib import Path

from .benchmarks import parse_mcqa, render_text
from .common import atomic_text, digest, now, stage_lock, write_json, write_jsonl
from .runner import load_run, prediction_map


def _integer(value, minimum=0):
    return type(value) is int and value >= minimum


def _duration(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _inspect_run(run_dir):
    root = Path(run_dir).resolve()
    if (root / ".generate.lock").exists():
        raise ValueError("Generation is still locked; inspect the run after the writer stops")
    manifest, records, _ = load_run(root)
    predictions = prediction_map(root, manifest)
    config = manifest["signature"]["config"]
    local = config["backend"] == "qwen3_vl_hf"
    reviews, batches = [], defaultdict(list)
    issues, tasks, finished = Counter(), {}, Counter()
    pairs = defaultdict(list)
    for record in records:
        task = f"{record['benchmark']}/{record['task']}"
        counts = tasks.setdefault(task, Counter())
        for index in range(manifest["responses_per_question"]):
            counts["expected"] += 1
            key = (record["id"], index)
            row = predictions.get(key)
            flags = []
            status = "missing" if row is None else row["status"]
            counts[status] += 1
            if status != "ok":
                flags.append(status)
            else:
                response = row["response"]
                if not response.strip():
                    flags.append("empty_response")
                reason = row.get("finish_reason")
                finished[str(reason)] += 1
                if reason == "length":
                    flags.append("truncated")
                elif reason != "stop":
                    flags.append("unexpected_finish_reason")
                if record["task"] == "mcqa" and parse_mcqa(response, record["options"]) is None:
                    flags.append("invalid_mcqa")
                if (row.get("model_fingerprint") != manifest["signature"]["target_fingerprint"] or
                        row.get("prompt_sha256") != digest(render_text(record)) or
                        row.get("protocol_id") != config["protocol_id"]):
                    flags.append("prediction_identity_mismatch")
                if not _integer(row.get("actual_batch_size"), 1) or row["actual_batch_size"] > config["batch_size"]:
                    flags.append("invalid_batch_size")
                if not isinstance(row.get("batch_id"), str) or not row["batch_id"] or not _duration(row.get("batch_wall_ms")):
                    flags.append("missing_batch_timing")
                elif _integer(row.get("actual_batch_size"), 1):
                    batches[row["batch_id"]].append(row)
                if local:
                    grids = row.get("image_grid_thw")
                    if (row.get("pixel_values_present") is not bool(record["media_paths"]) or row.get("image_count") != len(record["media_paths"]) or
                            not isinstance(grids, list) or len(grids) != len(record["media_paths"]) or any(
                                not isinstance(grid, list) or len(grid) != 3 or any(not _integer(n, 1) for n in grid) for grid in grids)):
                        flags.append("invalid_visual_inputs")
                    input_tokens, width, padding = (row.get(name) for name in ("input_tokens", "padded_input_tokens", "pad_tokens"))
                    if (not _integer(input_tokens, 1) or not _integer(width, 1) or not _integer(padding) or
                            width != input_tokens + padding):
                        flags.append("invalid_padding")
                    if not _integer(row.get("output_tokens"), 1) or row["output_tokens"] > config["max_new_tokens"]:
                        flags.append("invalid_output_tokens")
                    if (not _integer(row.get("peak_cuda_allocated_bytes"), 1) or
                            not _integer(row.get("peak_cuda_reserved_bytes"), 1) or
                            row["peak_cuda_reserved_bytes"] < row["peak_cuda_allocated_bytes"]):
                        flags.append("missing_memory_measurement")
            issues.update(flags)
            reviews.append({
                "id": record["id"], "response_index": index, "task": task, "pair_id": record["pair_id"],
                "query": record["query"], "media_paths": record["media_paths"], "options": record.get("options"),
                "status": status, "response": row.get("response") if row else None,
                "parsed_mcqa": parse_mcqa(row["response"], record["options"]) if status == "ok" and record["task"] == "mcqa" else None,
                "finish_reason": row.get("finish_reason") if row else None,
                "actual_batch_size": row.get("actual_batch_size") if row else None,
                "input_tokens": row.get("input_tokens") if row else None,
                "output_tokens": row.get("output_tokens") if row else None,
                "pad_tokens": row.get("pad_tokens") if row else None,
                "image_grid_thw": row.get("image_grid_thw") if row else None,
                "error": row.get("error") if row else None, "flags": flags,
            })
            if record["pair_id"]:
                pairs[(record["pair_id"], index)].append(status == "ok")
    batch_sizes = Counter()
    milliseconds, measured_requests, measured_tokens, tokens_known = 0.0, 0, 0, True
    for group in batches.values():
        first = group[0]
        if (first["actual_batch_size"] < len(group) or any(
                row["actual_batch_size"] != first["actual_batch_size"] or row["batch_wall_ms"] != first["batch_wall_ms"] for row in group)):
            issues["inconsistent_batch_membership"] += 1
            continue
        if first["actual_batch_size"] > len(group):
            # A crash may interrupt appending a completed batch. Do not invent its missing timing members.
            issues["partial_batch_telemetry"] += 1
            continue
        batch_sizes[str(len(group))] += 1
        milliseconds += first["batch_wall_ms"]
        measured_requests += len(group)
        for row in group:
            if _integer(row.get("output_tokens")):
                measured_tokens += row["output_tokens"]
            else:
                tokens_known = False
    if manifest.get("oom_batch_splits", 0):
        issues["oom_batch_splits"] = manifest["oom_batch_splits"]
    expected = len(reviews)
    good = sum(row["status"] == "ok" for row in reviews)
    complete = good == expected and expected > 0
    fatal = {"missing", "error", "empty_response", "prediction_identity_mismatch", "invalid_batch_size",
             "invalid_visual_inputs", "invalid_padding", "invalid_output_tokens", "inconsistent_batch_membership"}
    status = "failed" if any(issues[key] for key in fatal) or not complete else "needs_review" if issues else "passed"
    seconds = milliseconds / 1000
    peaks = {name: max((row.get(name, 0) for row in predictions.values() if _integer(row.get(name))), default=0) or None
             for name in ("peak_cuda_allocated_bytes", "peak_cuda_reserved_bytes")}
    report = {
        "schema_version": 1, "created_at": now(), "run_fingerprint": manifest["run_fingerprint"],
        "status": status, "complete": complete, "is_mock": manifest["is_mock"], "backend": config["backend"],
        "real_local_generation_passed": status == "passed" and local and not manifest["is_mock"],
        "scope": "Generation integrity only; no safety or capability judgment, and no API calls.",
        "expected_requests": expected, "successful_requests": good,
        "error_requests": sum(row["status"] == "error" for row in reviews),
        "missing_requests": sum(row["status"] == "missing" for row in reviews),
        "selected_mssbench_pairs": len(pairs),
        "complete_mssbench_pairs": sum(len(values) == 2 and all(values) for values in pairs.values()),
        "tasks": {key: dict(value) for key, value in tasks.items()},
        "issue_counts": {key: value for key, value in sorted(issues.items()) if value},
        "finish_reasons": dict(finished), "configured_batch_size": config["batch_size"],
        "performance": {
            "actual_batch_counts": dict(batch_sizes), "measured_successful_requests": measured_requests,
            "successful_batch_wall_seconds": round(seconds, 6),
            "requests_per_second": measured_requests / seconds if seconds else None,
            "output_tokens_per_second": measured_tokens / seconds if seconds and tokens_known else None,
            "timing_scope": "Successful backend calls only, including preprocessing; excludes model load and failed/OOM attempts.",
            **peaks,
        },
    }
    return report, reviews, manifest, records, predictions


def _write_review(output, rows):
    write_jsonl(output / "review.jsonl", rows)
    stream = io.StringIO(newline="")
    fields = list(rows[0]) if rows else ["id", "response_index", "flags"]
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow({key: " | ".join(map(str, value)) if isinstance(value, list) else value for key, value in row.items()})
    atomic_text(output / "review.csv", stream.getvalue())


def validate_run(run_dir):
    root = Path(run_dir).resolve()
    with stage_lock(root, "validate"):
        report, reviews, _, _, _ = _inspect_run(root)
        output = root / "checks"
        write_json(output / "validation.json", report)
        _write_review(output, reviews)
        lines = ["# Generation checks", "", f"Status: **{report['status']}**", "",
                 f"Mock: {report['is_mock']}. {report['scope']}", "",
                 f"Successful requests: {report['successful_requests']}/{report['expected_requests']}.",
                 f"Complete MSSBench pairs: {report['complete_mssbench_pairs']}/{report['selected_mssbench_pairs']}.", "",
                 "| Issue | Count |", "| --- | ---: |"]
        lines.extend(f"| {key} | {value} |" for key, value in report["issue_counts"].items())
        if not report["issue_counts"]:
            lines.append("| No issues detected | 0 |")
        lines.extend(["", "Inspect query, image paths, responses and flags in review.jsonl / review.csv.",
                      "Timing and peak memory are recorded in validation.json. Timing excludes model load and failed attempts.", ""])
        atomic_text(output / "validation.md", "\n".join(lines))
    return report


def compare_batches(before, after, output):
    if Path(before).resolve() == Path(after).resolve():
        raise ValueError("Use distinct batch-1 and batched run directories")
    left, left_rows, first, records, first_predictions = _inspect_run(before)
    right, right_rows, second, _, second_predictions = _inspect_run(after)
    first_config, second_config = (manifest["signature"]["config"] for manifest in (first, second))
    if first_config["do_sample"] or second_config["do_sample"]:
        raise ValueError("Batch comparison requires greedy generation")
    if first_config["batch_size"] != 1 or second_config["batch_size"] <= 1:
        raise ValueError("Compare batch_size=1 before against batch_size>1 after")
    for name in first["signature"].keys() | second["signature"].keys():
        a, b = first["signature"].get(name), second["signature"].get(name)
        if name == "config":
            a, b = ({key: value for key, value in config.items() if key != "batch_size"}
                    for config in (first_config, second_config))
        if a != b:
            raise ValueError(f"Batch comparison requires the same {name}; only batch_size may differ")
    if first["is_mock"] != second["is_mock"]:
        raise ValueError("Cannot compare mock and real runs")
    changes, counts = [], Counter()
    record_map = {row["id"]: row for row in records}
    for a, b in zip(left_rows, right_rows):
        key = (a["id"], a["response_index"])
        row_a, row_b = first_predictions.get(key), second_predictions.get(key)
        flags = []
        if a["status"] != "ok" or b["status"] != "ok":
            flags.append("incomplete_response")
        else:
            if a["response"] != b["response"]:
                flags.append("text_changed")
            if " ".join(a["response"].split()) != " ".join(b["response"].split()):
                flags.append("normalized_text_changed")
            for name in ("parsed_mcqa", "finish_reason", "input_tokens", "output_tokens", "image_grid_thw"):
                if a[name] != b[name]:
                    flags.append(name + "_changed")
            if row_a["seed"] != row_b["seed"] or row_a["prompt_sha256"] != row_b["prompt_sha256"]:
                flags.append("request_identity_changed")
        counts.update(flags)
        if flags:
            record = record_map[a["id"]]
            changes.append({"id": a["id"], "response_index": a["response_index"], "task": a["task"],
                            "query": record["query"], "media_paths": record["media_paths"], "flags": flags,
                            "before_response": a["response"], "after_response": b["response"],
                            "before_input_tokens": a["input_tokens"], "after_input_tokens": b["input_tokens"],
                            "before_grids": a["image_grid_thw"], "after_grids": b["image_grid_thw"]})
    actual = right["performance"]["actual_batch_counts"]
    batch_observed = int(actual.get(str(second_config["batch_size"]), 0)) > 0
    passed = left["status"] == right["status"] == "passed" and not changes and batch_observed
    report = {
        "schema_version": 1, "created_at": now(), "status": "passed" if passed else "needs_review",
        "is_mock": first["is_mock"], "before_fingerprint": first["run_fingerprint"], "after_fingerprint": second["run_fingerprint"],
        "expected_requests": left["expected_requests"], "changed_requests": len(changes), "change_counts": dict(counts),
        "configured_batch_observed": batch_observed, "before_validation": left, "after_validation": right,
        "real_local_batch_check_passed": passed and left["real_local_generation_passed"] and right["real_local_generation_passed"],
        "scope": "Same greedy requests with only batch_size changed. Differences require review; they are not automatically model bugs.",
    }
    output = Path(output).resolve()
    with stage_lock(output, "compare-batches"):
        write_json(output / "comparison.json", report)
        _write_review(output, changes)
        lines = ["# Batch comparison", "", f"Status: **{report['status']}**", "",
                 f"Mock: {report['is_mock']}. {report['scope']}", "",
                 f"Changed requests: {len(changes)}/{left['expected_requests']}.",
                 f"Configured batch {second_config['batch_size']} observed: {batch_observed}.",
                 f"Before validation: {left['status']}. After validation: {right['status']}.", "",
                 "Review differences in review.jsonl / review.csv. Read comparison.json for timings and memory.", ""]
        atomic_text(output / "comparison.md", "\n".join(lines))
    return report
