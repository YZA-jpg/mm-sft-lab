"""Audited WebP MIME repair; preserve frozen evaluation_v2 code and successful requests."""

import argparse
import json
import mimetypes
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation_v2 import cli
from evaluation_v2.common import digest, file_hash, latest, now, read_json, read_journal, write_json
from evaluation_v2.config import generation_parameters
from evaluation_v2.prompts import target_request
from evaluation_v2.runner import generation_jobs, load_run


def prepare_recovery(run_dir):
    """Validate the specific pre-HTTP error and save a repair receipt; never call a model."""
    root = Path(run_dir).resolve()
    for lock in (root / ".pipeline.lock", root / "generation/.stage.lock"):
        if lock.exists():
            raise ValueError(f"Run is locked; inspect its owner before recovery: {lock}")
    manifest, records, data = load_run(root)
    config, fingerprint = manifest["signature"]["config"], manifest["run_fingerprint"]
    rows = latest(read_journal(root / "generation/results.jsonl"))
    if any(r.get("run_fingerprint") != fingerprint for r in rows.values()):
        raise ValueError("Journal belongs to another run")
    jobs = {j["job_id"]: j for j in generation_jobs(records, config)}
    if set(rows) - set(jobs):
        raise ValueError("Unexpected generation jobs")
    failures = [r for r in rows.values() if r["status"] != "ok"]
    if not failures:
        raise ValueError("No failed job to repair; inspect status instead of restarting successes")
    events = read_journal(root / "generation/transport-attempts.jsonl")
    for row in failures:
        error = row.get("error", "")
        record = jobs[row["job_id"]]["record"]
        if (row.get("error_kind") != "local_preparation_error"
                or "TypeError" not in error or "NoneType" not in error
                or "concatenate str" not in error
                or not any(Path(name).suffix.lower() == ".webp" for name in record["images"])):
            raise ValueError(f"Unrelated failure; no automatic resume: {row['job_id']}: {error}")
        if any(e["job_id"] == row["job_id"] for e in events):
            raise ValueError("Expected a pre-HTTP MIME error, but failed job has transport attempts")

    before = mimetypes.guess_type("probe.webp")[0]
    mimetypes.add_type("image/webp", ".webp", strict=True)
    images = {name for r in records for name in r["images"]}
    unknown = sorted(name for name in images if mimetypes.guess_type(name)[0] is None)
    if unknown:
        raise ValueError("Other unsupported MIME suffixes remain: " + ", ".join(unknown[:5]))

    data_root = Path(manifest["data_root"])
    for row in failures:
        request = target_request(jobs[row["job_id"]]["record"], data_root, data)
        for item in request["messages"][0]["content"]:
            if item["type"] == "image_url" and not item["image_url"]["url"].startswith("data:image/"):
                raise ValueError("Expected an image data URI")

    successful = [r for r in rows.values() if r["status"] == "ok"]
    for index, row in enumerate(successful, 1):
        request = target_request(jobs[row["job_id"]]["record"], data_root, data)
        payload = {"model": config["target"]["model"], **request, **generation_parameters(config)}
        if config["execution"]["generation_stream"]:
            payload.update(stream=True, stream_options={"include_usage": True, "continuous_usage_stats": True})
        if digest(payload) != row.get("request_sha256"):
            raise ValueError("Previously successful request hash changed: " + row["job_id"])
        if digest(row["text"]) != row.get("response_sha256"):
            raise ValueError("Previously successful response hash changed: " + row["job_id"])
        if index % 100 == 0:
            print(f"Verified successful request hashes: {index}/{len(successful)}", flush=True)

    stamp = now()
    output = root / "recovery" / ("webp-" + stamp.replace(":", "-"))
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "resolved-config.json", config)
    known_usage = [r for r in successful if type((r.get("usage") or {}).get("completion_tokens")) is int]
    receipt = {
        "created_at": stamp, "scope": "MIME metadata compatibility repair; no model or API call yet",
        "run_dir": str(root), "run_fingerprint": fingerprint, "is_mock": manifest["is_mock"],
        "code_sha256": manifest["signature"]["code_sha256"],
        "recovery_script_sha256": file_hash(Path(__file__)),
        "webp_mime_before": before, "webp_mime_after": mimetypes.guess_type("probe.webp")[0],
        "failed_jobs": [{"job_id": r["job_id"], "error": r.get("error")} for r in failures],
        "successful_preserved": len(successful),
        "verified_successful_request_hashes": len(successful),
        "successful_snapshot_sha256": digest(sorted((r["job_id"], r["response_sha256"]) for r in successful)),
        "generation_journal_sha256_before_resume": file_hash(root / "generation/results.jsonl"),
        "expected_jobs": len(jobs), "pending_jobs_including_repair": len(jobs) - len(successful),
        "previous_job_pid_record": (root / "job.pid").read_text().strip() if (root / "job.pid").exists() else None,
        "successful_output_usage": {
            "known_usage_jobs": len(known_usage),
            "completion_tokens": sum(r["usage"]["completion_tokens"] for r in known_usage),
            "length_jobs": sum(r.get("finish_reason") == "length" for r in successful),
            "completion_tokens_from_length": sum(r["usage"]["completion_tokens"] for r in known_usage
                                                 if r.get("finish_reason") == "length"),
            "scope": "Final successful outputs only; partial failed-attempt costs remain in original journals.",
        },
        "config": {"generation": config["generation"], "repeats": config["repeats"],
                   "execution": config["execution"]},
        "note": "Original image bytes, prompts, journal history and frozen evaluator files remain unchanged.",
    }
    path = output / "receipt.json"
    write_json(path, receipt)
    return path, receipt


def resume_arguments(run_dir, config_path):
    root = Path(run_dir).resolve()
    manifest = read_json(root / "manifest.json")
    model = manifest["signature"]["model_identity"]
    runtime = model["serving_runtime"]
    argv = ["run", "--config", str(config_path), "--data-root", manifest["data_root"],
            "--model-path", model["path"], "--model-label", manifest["model_label"],
            "--run-dir", str(root), "--gpu", runtime["gpu"],
            "--vllm-python", runtime["python"], "--resume", "--generate-only"]
    for flag, key in (("--gpu-memory-utilization", "memory_utilization"),
                      ("--target-kv-cache-gib", "target_kv_cache_gib"),
                      ("--wildguard-kv-cache-gib", "wildguard_kv_cache_gib")):
        if runtime.get(key) is not None:
            argv += [flag, str(runtime[key])]
    if model.get("adapter"):
        argv += ["--adapter-path", model["adapter"]["path"]]
    if model.get("wildguard_identity"):
        argv += ["--wildguard-path", model["wildguard_identity"]["path"]]
    return argv


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--check-only", action="store_true", help="Validate and save an audit receipt; no GPU/model/API")
    args = parser.parse_args(argv)
    print("Checking frozen run, original error and saved requests; no GPU/API calls yet.", flush=True)
    try:
        receipt_path, receipt = prepare_recovery(args.run_dir)
    except (ValueError, OSError, KeyError) as error:
        print(json.dumps({"status": "blocked", "error": f"{type(error).__name__}: {error}"},
                         ensure_ascii=False, indent=2), flush=True)
        return 2
    print(json.dumps({"status": "compatibility_checked_no_model_started",
                      "receipt": str(receipt_path), **receipt}, ensure_ascii=False, indent=2), flush=True)
    if args.check_only:
        return 0
    return cli.main(resume_arguments(args.run_dir, receipt_path.parent / "resolved-config.json"))


if __name__ == "__main__":
    raise SystemExit(main())

