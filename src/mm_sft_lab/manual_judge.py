"""Human review contracts; judgments stay tied to exact generated responses."""

from pathlib import Path

from .common import digest, file_hash, now, read_jsonl, stage_lock, write_json, write_jsonl
from .judge import MSS_RUBRIC, SIUO_RUBRIC, WILD_RUBRIC, needs_judge, parse_labels
from .runner import load_run, prediction_map


def export_review(run_dir, output):
    run_dir, output = Path(run_dir).resolve(), Path(output).resolve()
    if (run_dir / ".generate.lock").exists():
        raise ValueError("Finish generation before exporting review")
    if output.exists():
        raise ValueError("Review file already exists; choose a new file to preserve edited labels")
    manifest, records, labels = load_run(run_dir)
    predictions = prediction_map(run_dir, manifest)
    rows = []
    for record in records:
        if not needs_judge(record):
            continue
        for index in range(manifest["responses_per_question"]):
            prediction = predictions.get((record["id"], index))
            if not prediction or prediction["status"] != "ok":
                continue
            rubric = SIUO_RUBRIC if record["benchmark"] == "siuo" else WILD_RUBRIC if record["benchmark"] == "wildjailbreak" else MSS_RUBRIC
            rows.append({"id": record["id"], "response_index": index, "run_fingerprint": manifest["run_fingerprint"],
                         "response_sha256": prediction["response_sha256"], "benchmark": record["benchmark"], "task": record["task"],
                         "query": record["query"], "media_paths": record["media_paths"], "data_root": manifest["data_root"],
                         "reference": labels[record["id"]], "response": prediction["response"],
                         "finish_reason": prediction.get("finish_reason"), "rubric": rubric, "labels": None})
    write_jsonl(output, rows)
    return {"status": "exported", "requests": len(rows), "output": str(output), "is_mock": manifest["is_mock"]}


def import_manual(run_dir, source, judge_dir, *, reviewer="human", protocol="manual_lab_rubric_v1"):
    run_dir, source, judge_dir = Path(run_dir).resolve(), Path(source).resolve(), Path(judge_dir).resolve()
    if (run_dir / ".generate.lock").exists():
        raise ValueError("Finish generation before importing labels")
    if not reviewer.strip() or not protocol.strip():
        raise ValueError("Reviewer and protocol must be nonempty")
    manifest, records, labels = load_run(run_dir)
    record_map = {row["id"]: row for row in records}
    predictions = prediction_map(run_dir, manifest)
    signature = digest({"kind": "manual", "protocol": protocol, "reviewer": reviewer,
                        "rubrics": [SIUO_RUBRIC, MSS_RUBRIC, WILD_RUBRIC]})
    converted, seen = [], set()
    for row in read_jsonl(source):
        key = (row.get("id"), row.get("response_index"))
        if type(row.get("response_index")) is not int or key in seen:
            raise ValueError("Duplicate or invalid manual review key")
        seen.add(key)
        record = record_map.get(row.get("id"))
        prediction = predictions.get(key)
        if not record or not needs_judge(record) or not prediction or prediction["status"] != "ok":
            raise ValueError(f"Unexpected manual judgment: {key}")
        if row.get("run_fingerprint") != manifest["run_fingerprint"] or row.get("response_sha256") != prediction["response_sha256"]:
            raise ValueError(f"Stale manual response identity: {key}")
        if row.get("response") != prediction["response"] or row.get("query") != record["query"]:
            raise ValueError("Edit labels only; do not change the question or response")
        if row.get("reference") != labels[record["id"]] or row.get("media_paths") != record["media_paths"]:
            raise ValueError("Edit labels only; references and image paths must stay unchanged")
        if row.get("labels") is None:
            continue
        import json
        parsed = parse_labels(json.dumps(row["labels"]), record["benchmark"])
        converted.append({"id": key[0], "response_index": key[1], "status": "ok", "attempt": 1,
                          "response_sha256": prediction["response_sha256"], "judge_signature": signature,
                          "labels": parsed, "raw_judgment": row["labels"], "origin": "manual", "recorded_at": now()})
    if not converted:
        raise ValueError("No completed manual labels; labels:null stays pending")
    with stage_lock(judge_dir, "judge"):
        if any(p.name != ".judge.lock" for p in judge_dir.iterdir()):
            raise ValueError("Use a new judge directory for each manual label snapshot")
        expected = sum(needs_judge(row) for row in records) * manifest["responses_per_question"]
        result = {"schema_version": 1, "created_at": now(), "run_fingerprint": manifest["run_fingerprint"],
                  "judge_signature": signature, "judge_protocol_id": protocol, "identity": {"reviewer": reviewer},
                  "is_mock": False, "source_file_sha256": file_hash(source), "expected_requests": expected,
                  "successful_requests": len(converted), "pending_requests": expected - len(converted),
                  "status": "complete" if len(converted) == expected else "incomplete"}
        write_jsonl(judge_dir / "judgments.jsonl", converted)
        write_json(judge_dir / "manifest.json", result)
    return result


def export_textvqa_test(run_dir, output):
    manifest, records, labels = load_run(run_dir)
    if (Path(run_dir) / ".generate.lock").exists() or manifest["responses_per_question"] != 1:
        raise ValueError("Finish a single-response inference run before exporting Test")
    predictions = prediction_map(run_dir, manifest)
    selected = [row for row in records if row["benchmark"] == "textvqa" and row["task"] == "test"]
    if not selected:
        raise ValueError("Run contains no TextVQA Test requests")
    output_rows = []
    for record in selected:
        prediction = predictions.get((record["id"], 0))
        if not prediction or prediction["status"] != "ok":
            raise ValueError("TextVQA Test generation has missing or failed requests")
        output_rows.append({"question_id": labels[record["id"]]["question_id"], "answer": prediction["response"].strip()})
    if Path(output).exists():
        raise ValueError("Submission file already exists; choose another output")
    write_json(output, output_rows)
    return {"status": "exported", "requests": len(output_rows), "output": str(Path(output).resolve()),
            "official_score": "pending", "is_mock": manifest["is_mock"]}
