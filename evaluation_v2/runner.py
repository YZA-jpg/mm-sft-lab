"""Fair bounded concurrency, durable stage journals, and independently resumable phases."""

import itertools
import os
import sys
import threading
import time
from collections import Counter, defaultdict, deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from .common import (append, code_identity, digest, file_hash, latest, lock, now, read_json,
                     read_journal, write_json, write_jsonl)
from .config import generation_parameters
from .data import load_dataset
from .grading import grade_result
from .prompts import judge_request, target_request
from .transport import Client


def initialize(data_root, run_dir, config, model_identity, *, resume=False):
    data_root, run_dir = Path(data_root).resolve(), Path(run_dir).resolve()
    rows, data = load_dataset(data_root, config)
    signature = {"protocol_version": 2, "config": config, "model_identity": model_identity,
                 "data_manifest_sha256": file_hash(data_root / "manifest.json"),
                 "selection_sha256": digest(rows), "code_sha256": code_identity()["sha256"]}
    identity = digest(signature)
    with lock(run_dir, "initialize"):
        path = run_dir / "manifest.json"
        if path.exists():
            saved = read_json(path)
            if not resume or saved["run_fingerprint"] != identity:
                raise ValueError("Existing run differs; resume requires identical model/data/protocol/concurrency/code")
        else:
            if resume:
                raise ValueError("Cannot resume without a run manifest")
            write_jsonl(run_dir / "selection.jsonl", rows)
            write_json(run_dir / "dataset-manifest.json", data)
            saved = {"schema_version": 2, "created_at": now(), "run_fingerprint": identity,
                     "signature": signature, "data_root": str(data_root), "model_label": model_identity.get("label", "target"),
                     "is_mock": bool(data["is_mock"] or model_identity.get("kind") == "mock"),
                     "snapshot_sha256": {n: file_hash(run_dir / n) for n in ("selection.jsonl", "dataset-manifest.json")}}
            write_json(path, saved)
    return saved


def load_run(run_dir):
    run_dir = Path(run_dir).resolve()
    manifest = read_json(run_dir / "manifest.json")
    if digest(manifest["signature"]) != manifest["run_fingerprint"]:
        raise ValueError("Run fingerprint changed")
    if code_identity()["sha256"] != manifest["signature"]["code_sha256"]:
        raise ValueError("Evaluation v2 code changed; use original code or create a new run")
    for name, sha in manifest["snapshot_sha256"].items():
        if file_hash(run_dir / name) != sha:
            raise ValueError("Run data snapshot changed")
    data_root = Path(manifest["data_root"])
    if file_hash(data_root / "manifest.json") != manifest["signature"]["data_manifest_sha256"]:
        raise ValueError("Prepared dataset manifest changed")
    rows, data = load_dataset(data_root, manifest["signature"]["config"])
    if digest(rows) != manifest["signature"]["selection_sha256"]:
        raise ValueError("Selection changed")
    return manifest, rows, data


def generation_jobs(rows, config):
    return [{"job_id": f"{row['id']}#g{index}", "sample_id": row["id"], "trial": index,
             "benchmark": row["benchmark"], "split": row["split"], "record": row}
            for row in rows for index in range(config["repeats"][row["benchmark"]])]


def fair_order(jobs):
    groups = defaultdict(lambda: defaultdict(deque))
    for job in jobs:
        groups[job["benchmark"]][job["split"]].append(job)
    active = deque(deque(splits.values()) for splits in groups.values())
    while active:
        splits = active.popleft()
        group = splits.popleft()
        yield group.popleft()
        if group:
            splits.append(group)
        if splits:
            active.append(splits)


def _cached(path, fingerprint, *, repair=False):
    rows = latest(read_journal(path, repair=repair))
    for row in rows.values():
        if row.get("run_fingerprint") != fingerprint:
            raise ValueError("Journal belongs to another run")
    return rows


def stage(run_dir, name, *, client_factory=Client, retry_recoverable=False):
    if name not in ("generation", "deepseek", "wildguard"):
        raise ValueError("Unknown phase")
    run_dir = Path(run_dir).resolve()
    manifest, records, data = load_run(run_dir)
    config, root = manifest["signature"]["config"], Path(manifest["data_root"])
    role = "target" if name == "generation" else name
    jobs = generation_jobs(records, config)
    generations = {}
    if name != "generation":
        if (run_dir / "generation/.stage.lock").exists():
            raise ValueError("Finish generation before judging")
        generations = _cached(run_dir / "generation/results.jsonl", manifest["run_fingerprint"])
        generated_jobs, jobs = jobs, []
        for job in generated_jobs:
            prediction = generations.get(job["job_id"])
            benchmark = job["benchmark"]
            selected = benchmark == "wildjailbreak" if name == "wildguard" else benchmark not in ("wildjailbreak", "mmstar")
            if not selected or prediction is None or prediction["status"] != "ok":
                continue
            if digest(prediction["text"]) != prediction["response_sha256"]:
                raise ValueError("Generated response changed")
            for index in range(config["mmvet_judge_repeats"] if benchmark == "mmvet" else 1):
                jobs.append({**job, "generation_id": job["job_id"], "job_id": job["job_id"] + f"#j{index}",
                             "judge_repeat": index, "prediction": prediction})
    directory = run_dir / name
    with lock(directory, "stage"):
        previous = _cached(directory / "results.jsonl", manifest["run_fingerprint"], repair=True)
        job_map = {j["job_id"]: j for j in jobs}
        if set(previous) - set(job_map):
            raise ValueError("Cached results outside current phase inputs")
        for key, row in previous.items():
            if name == "generation":
                if row["status"] == "ok" and digest(row["text"]) != row["response_sha256"]:
                    raise ValueError("Cached generation changed")
            elif row["input_response_sha256"] != job_map[key]["prediction"]["response_sha256"]:
                raise ValueError("Cached judgment is tied to a different response")
        events = read_journal(directory / "transport-attempts.jsonl", repair=True)
        offsets = Counter()
        for event in events:
            if event["job_id"] not in job_map or event["run_fingerprint"] != manifest["run_fingerprint"]:
                raise ValueError("Unexpected transport attempt event")
            if type(event["number"]) is not int or event["number"] != offsets[event["job_id"]] + 1 or event["number"] > 6:
                raise ValueError("Invalid transport retry history")
            offsets[event["job_id"]] = event["number"]
        recoverable = {"authentication_or_configuration_error", "local_preparation_error"}
        remaining = [j for j in jobs if j["job_id"] not in previous or
                     (retry_recoverable and previous[j["job_id"]].get("error_kind") in recoverable)]
        todo = iter(fair_order(remaining))
        client = client_factory(config[role], config["execution"]) if remaining else None
        count_key = "generation_concurrency" if name == "generation" else name + "_concurrency"
        concurrency = config["execution"][count_key]
        event_lock, cancelled = threading.Lock(), threading.Event()
        session_id, session_started, fatal = now(), time.time(), False
        timing = time.perf_counter()
        submitted = 0
        append(directory / "sessions.jsonl", {"event": "started", "session_id": session_id,
                "started_epoch": session_started, "concurrency": concurrency})

        def work(job, queued):
            entered = time.time()
            started = time.perf_counter()
            row = {k: v for k, v in job.items() if k not in ("record", "prediction")}
            row.update(run_fingerprint=manifest["run_fingerprint"], attempt=previous.get(job["job_id"], {}).get("attempt", 0) + 1,
                       session_id=session_id, recorded_at=now(), queue_seconds=entered - queued,
                       job_started_epoch=entered)

            def event(value):
                with event_lock:
                    append(directory / "transport-attempts.jsonl", {**value, "job_id": job["job_id"],
                           "run_fingerprint": manifest["run_fingerprint"], "session_id": session_id})
            try:
                if name == "generation":
                    request = target_request(job["record"], root, data)
                else:
                    _, request = judge_request(job["record"], job["prediction"]["text"], root, data)
                    row["input_response_sha256"] = job["prediction"]["response_sha256"]
                row["preparation_seconds"] = time.perf_counter() - started
                transport = generation_parameters(config) if name == "generation" else {}
                if name == "generation" and config["execution"]["generation_stream"]:
                    stream_directory = directory / "streams" / digest(job["job_id"])
                    stream_directory.mkdir(parents=True, exist_ok=True)
                    identity = {"job_id": job["job_id"], "run_fingerprint": manifest["run_fingerprint"]}
                    identity_path = stream_directory / "identity.json"
                    if identity_path.exists() and read_json(identity_path) != identity:
                        raise ValueError("Stream capture identity differs")
                    write_json(identity_path, identity)
                    transport.update(stream=True, stream_directory=stream_directory, cancel_event=cancelled)
                result = client.complete(**request, attempt_offset=offsets[job["job_id"]], on_attempt=event, **transport)
                row.update(result if name == "generation" else grade_result(job["benchmark"], result))
            except Exception as error:
                row.update(status="error", error_kind="local_preparation_error", error=f"{type(error).__name__}: {error}",
                           text=None, labels=None, fatal=True, attempts=[])
                if name != "generation":
                    row["input_response_sha256"] = job["prediction"]["response_sha256"]
            row.update(job_finished_epoch=time.time(), job_wall_seconds=time.perf_counter() - started)
            return row

        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            pending = {}

            def fill():
                nonlocal submitted
                while not fatal and len(pending) < concurrency:
                    job = next(todo, None)
                    if job is None:
                        break
                    pending[pool.submit(work, job, time.time())] = job["job_id"]
                    submitted += 1
            try:
                fill()
                while pending:
                    done, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        pending.pop(future)
                        row = future.result()
                        append(directory / "results.jsonl", row)
                        previous[row["job_id"]] = row
                        fatal = fatal or row.get("fatal", False)
                        print(f"{name} [{len(previous)}/{len(jobs)}] {row['job_id']}: {row['status']} {row.get('error_kind') or ''}", flush=True)
                    fill()
            except BaseException:
                cancelled.set()
                raise
        session = {"event": "completed", "session_id": session_id, "started_epoch": session_started, "finished_epoch": time.time(),
                   "wall_seconds": time.perf_counter() - timing, "new_jobs": submitted, "concurrency": concurrency}
        append(directory / "sessions.jsonl", session)
        result = {"schema_version": 2, "run_fingerprint": manifest["run_fingerprint"], "updated_at": now(),
                  "phase": name, "expected_jobs": len(jobs), "handled_jobs": len(previous),
                  "successful_jobs": sum(r["status"] == "ok" for r in previous.values()),
                  "error_jobs": sum(r["status"] != "ok" for r in previous.values()),
                  "unattempted_jobs": len(jobs) - len(previous), "fatal": fatal,
                  "processing_complete": len(jobs) == len(previous),
                  "scoring_complete": len(jobs) == len(previous) and all(r["status"] == "ok" for r in previous.values()),
                  "last_session": session}
        if name == "generation":
            result.update(parameters=generation_parameters(config),
                          finish_reasons=dict(Counter(r.get("finish_reason") for r in previous.values())),
                          truncated_jobs=sum(r["status"] == "ok" and r.get("finish_reason") == "length"
                                             for r in previous.values()))
        write_json(directory / "manifest.json", result)
    return result
