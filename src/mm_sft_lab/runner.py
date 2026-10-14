"""Immutable run manifests and append-only, restartable generation."""

import gc
import time
from pathlib import Path

from .backends import api_identity, make_backend
from .benchmarks import PROMPT_VERSION, render_text
from .common import append_jsonl, digest, file_hash, latest_attempts, now, read_json, read_jsonl, stage_lock, write_json, write_jsonl
from .data import load_prepared
from .preflight import code_identity, environment, model_identity


def load_run(run_dir):
    root = Path(run_dir).resolve()
    manifest = read_json(root / "manifest.json")
    if digest(manifest["signature"]) != manifest["run_fingerprint"]:
        raise ValueError("Run manifest fingerprint mismatch")
    for name, expected in manifest["snapshot_sha256"].items():
        if file_hash(root / name) != expected:
            raise ValueError(f"Run snapshot changed: {name}")
    records = read_jsonl(root / "selection.jsonl")
    labels = read_jsonl(root / "evaluation.jsonl")
    if digest(records) != manifest["signature"]["selection_sha256"] or digest(labels) != manifest["signature"]["labels_sha256"]:
        raise ValueError("Run selection or evaluation snapshot mismatch")
    return manifest, records, {row["id"]: row for row in labels}


def prediction_map(run_dir, manifest, *, repair_tail=False):
    result = latest_attempts(read_jsonl(Path(run_dir) / "predictions.jsonl", repair_tail=repair_tail))
    expected = {(row["id"], index) for row in read_jsonl(Path(run_dir) / "selection.jsonl")
                for index in range(manifest["responses_per_question"])}
    for key, row in result.items():
        if key not in expected or row.get("run_fingerprint") != manifest["run_fingerprint"]:
            raise ValueError(f"Unexpected prediction identity: {key}")
        if row["status"] == "ok" and (not isinstance(row.get("response"), str) or row.get("response_sha256") != digest(row["response"])):
            raise ValueError(f"Prediction response hash mismatch: {key}")
    return result


def generate(config, data_root, run_dir, *, resume=False, retry_errors=False, backend_factory=None):
    data_root, run_dir = Path(data_root).resolve(), Path(run_dir).resolve()
    records, labels, data_manifest = load_prepared(data_root, config)
    code = code_identity()
    target = ({"fingerprint": "mock_backend_v1"} if config["backend"] == "mock" else
              api_identity(config["api"]) if config["backend"] == "chat_api" else model_identity(config["model_path"]))
    signature = {
        "config": config, "selection_sha256": digest(records), "labels_sha256": digest(labels),
        "prepared_manifest_sha256": file_hash(data_root / "prepared/manifest.json"),
        "code_sha256": code["package_sha256"], "target_fingerprint": target["fingerprint"],
        "prompt_version": PROMPT_VERSION, "environment": environment(),
    }
    fingerprint = digest(signature)
    with stage_lock(run_dir, "generate"):
        path = run_dir / "manifest.json"
        if path.exists():
            if not resume:
                raise ValueError("Run exists; pass --resume or choose a new run directory")
            manifest, _, _ = load_run(run_dir)
            if manifest["run_fingerprint"] != fingerprint:
                raise ValueError("Run configuration, data, code, or target changed; choose a new run directory")
        else:
            if resume:
                raise ValueError("Cannot resume a run without a manifest")
            if any(path.name != ".generate.lock" for path in run_dir.iterdir()):
                raise ValueError("New run directory must be empty")
            write_json(run_dir / "resolved_config.json", config)
            write_jsonl(run_dir / "selection.jsonl", records)
            write_jsonl(run_dir / "evaluation.jsonl", labels)
            manifest = {
                "schema_version": 1, "created_at": now(), "status": "initialized",
                "run_fingerprint": fingerprint, "signature": signature, "code": code,
                "environment": environment(), "target": target, "data_root": str(data_root),
                "data_provenance": data_manifest["provenance_status"], "is_mock": config["backend"] == "mock",
                "responses_per_question": len(config["seeds"]),
                "oom_batch_splits": 0,
                "expected_requests": len(records) * len(config["seeds"]),
                "snapshot_sha256": {name: file_hash(run_dir / name) for name in (
                    "resolved_config.json", "selection.jsonl", "evaluation.jsonl")},
            }
            write_json(path, manifest)
        previous = prediction_map(run_dir, manifest, repair_tail=resume)
        work = []
        for row in records:
            for index, base_seed in enumerate(config["seeds"]):
                old = previous.get((row["id"], index))
                if old is None or (retry_errors and old["status"] == "error"):
                    seed = int(digest([row["id"], index, base_seed])[:8], 16) % (2 ** 31)
                    work.append((row, index, seed))
        started = time.perf_counter()
        try:
            if work:
                backend = (backend_factory or make_backend)(config)
                if backend.metadata.get("fingerprint") != target["fingerprint"]:
                    raise ValueError("Target identity changed while loading the backend")
                manifest["backend_metadata"] = backend.metadata
                manifest["status"] = "running"
                write_json(path, manifest)

                def run_batch(batch):
                    rows = [item[0] for item in batch]
                    batch_started = time.perf_counter()
                    try:
                        results = backend.generate(rows, data_root, config, [item[2] for item in batch])
                        if len(results) != len(batch) or any(not isinstance(result.get("response"), str) for result in results):
                            raise ValueError("Backend result count or text type is invalid")
                    except Exception as error:
                        if type(error).__name__ == "OutOfMemoryError" and len(batch) > 1 and config["backend"] == "qwen3_vl_hf":
                            manifest["oom_batch_splits"] = manifest.get("oom_batch_splits", 0) + 1
                            error.__traceback__ = None
                            gc.collect()
                            backend.torch.cuda.empty_cache()
                            middle = len(batch) // 2
                            run_batch(batch[:middle])
                            run_batch(batch[middle:])
                            return
                        text = (backend.client.safe_error(error) if hasattr(backend, "client") else f"{type(error).__name__}: {str(error)[:1000]}")
                        results = [{"status": "error", "error": text, "response": None} for _ in batch]
                        if len(batch) == 1 and type(error).__name__ == "OutOfMemoryError":
                            print("OOM at batch size 1: image budget is unchanged; request recorded as error", flush=True)
                        fatal = getattr(error, "status_code", None) in (400, 401, 403, 404)
                    else:
                        fatal = False
                    batch_wall_ms = round((time.perf_counter() - batch_started) * 1000, 3)
                    batch_id = digest([fingerprint, [(row["id"], index, previous.get((row["id"], index), {}).get("attempt", 0) + 1)
                                                     for row, index, _ in batch]])[:24]
                    for (row, index, seed), result in zip(batch, results):
                        key = (row["id"], index)
                        attempt = previous.get(key, {}).get("attempt", 0) + 1
                        value = {
                            **result, "id": row["id"], "response_index": index, "seed": seed,
                            "batch_id": batch_id, "actual_batch_size": len(batch), "batch_wall_ms": batch_wall_ms,
                            "attempt": attempt, "status": result.get("status", "ok"), "recorded_at": now(),
                            "run_fingerprint": fingerprint, "model_fingerprint": target["fingerprint"],
                            "protocol_id": config["protocol_id"], "prompt_sha256": digest(render_text(row)),
                            "response_sha256": digest(result["response"]) if result["response"] is not None else None,
                        }
                        append_jsonl(run_dir / "predictions.jsonl", value)
                        previous[key] = value
                        if value["status"] == "error":
                            append_jsonl(run_dir / "errors.jsonl", value)
                        print(f"[{len(previous)}/{manifest['expected_requests']}] {row['id']} #{index}: {value['status']}", flush=True)
                    if fatal:
                        raise ValueError("API rejected the request; check model, credentials, and parameter support, then resume with --retry-errors")

                consecutive_errors = 0
                for offset in range(0, len(work), config["batch_size"]):
                    run_batch(work[offset:offset + config["batch_size"]])
                    batch_keys = [(row["id"], index) for row, index, _ in work[offset:offset + config["batch_size"]]]
                    consecutive_errors = consecutive_errors + 1 if all(previous[key]["status"] == "error" for key in batch_keys) else 0
                    if consecutive_errors >= 3:
                        raise ValueError("Three consecutive batches failed; inspect errors.jsonl before resuming")
        finally:
            good = sum(row["status"] == "ok" for row in previous.values())
            manifest.update(status="complete" if good == manifest["expected_requests"] else "incomplete",
                            updated_at=now(), successful_requests=good,
                            error_requests=sum(row["status"] == "error" for row in previous.values()),
                            missing_requests=manifest["expected_requests"] - len(previous),
                            last_session_seconds=round(time.perf_counter() - started, 3))
            write_json(path, manifest)
    return manifest
