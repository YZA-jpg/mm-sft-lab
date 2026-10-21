"""Seven real SIUO/MSSBench requests by default; local vLLM only, no judge or decoding overrides."""

import argparse
import base64
import http.client
import json
import mimetypes
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections import Counter, defaultdict
from pathlib import Path

from start_vllm_model import (DEFAULT_SERVER, REPO, checkpoint_identity, digest, now,
                              owned_alive, read_json, sha_file, write_json)

sys.path.insert(0, str(REPO / "src"))
from mm_sft_lab.benchmarks import render_text
from mm_sft_lab.common import append_jsonl, read_jsonl, stage_lock
from mm_sft_lab.data import load_prepared

TASKS = ["siuo/gen", "mssbench/chat", "mssbench/embodied"]


def find_data_root(explicit=None):
    if explicit:
        return Path(explicit).resolve()
    previous = REPO / "runs/B0-full-20261016T091009Z/main/manifest.json"
    if previous.exists():
        candidate = Path(read_json(previous)["data_root"]).resolve()
        if (candidate / "prepared/manifest.json").is_file():
            return candidate
    candidates = {p.resolve() for p in (REPO / "data/step1-imported-v1", REPO / "data/step1-imported", REPO / "data/step1")
                  if (p / "prepared/manifest.json").is_file()}
    if len(candidates) != 1:
        raise ValueError("Cannot uniquely locate existing data; pass --data-root /path/to/the/imported/data")
    return candidates.pop()


def select_jobs(root, limit):
    records, _, manifest = load_prepared(root, {"require_locked_data": True, "tasks": TASKS, "per_task_limit": limit})
    if manifest.get("is_mock"):
        raise ValueError("This script requires real validated data, not a mock snapshot")
    jobs = [{"job_id": row["id"] + f"#g{trial}", "trial": trial, "record": row}
            for row in records for trial in range(3 if row["benchmark"] == "siuo" else 1)]
    return jobs, manifest


def make_payload(record, root, manifest, model_id):
    images = {row["path"]: row["sha256"] for row in manifest["images"]}
    if len(record["media_paths"]) != 1:
        raise ValueError("This smoke expects exactly one image per request")
    content = []
    for name in record["media_paths"]:
        path = (Path(root) / name).resolve()
        if not path.is_relative_to(Path(root).resolve()) or sha_file(path) != images[name]:
            raise ValueError("Image changed after data validation")
        media = mimetypes.guess_type(name)[0]
        if media not in ("image/png", "image/jpeg", "image/webp", "image/gif"):
            raise ValueError("Unsupported image media type")
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        content.append({"type": "image_url", "image_url": {"url": f"data:{media};base64,{encoded}"}})
    content.append({"type": "text", "text": render_text(record)})
    return {"model": model_id, "messages": [{"role": "user", "content": content}]}


def wait_for_server(receipt, seconds):
    url = urllib.parse.urlsplit(receipt["base_url"])
    if url.scheme != "http" or url.hostname != "127.0.0.1" or url.query or url.fragment or url.username:
        raise ValueError("Smoke requests must use the launcher's local server endpoint")
    started, last_print = time.monotonic(), -30.0
    while time.monotonic() - started < seconds:
        if not owned_alive(receipt):
            raise RuntimeError(f"Owned server is not running; inspect {receipt['log']}")
        try:
            with urllib.request.urlopen(receipt["base_url"] + "/models", timeout=5) as stream:
                values = json.load(stream)
            models = {item["id"]: item for item in values["data"]}
            if receipt["model_id"] not in models:
                raise ValueError("Expected model ID is absent; refusing another model")
            if Path(models[receipt["model_id"]].get("root", "")).resolve() != Path(receipt["model_path"]).resolve():
                raise ValueError("Served checkpoint root differs from the requested model")
            return time.monotonic() - started
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            elapsed = time.monotonic() - started
            if elapsed - last_print >= 30:
                print(f"Waiting for vLLM readiness: {elapsed:.0f}s; log: {receipt['log']}", flush=True)
                last_print = elapsed
            time.sleep(2)
    raise TimeoutError(f"Model not ready within {seconds}s; inspect {receipt['log']}")


def completion(base_url, payload, timeout, offset=0, on_attempt=None, *, opener=None, sleeper=None):
    if set(payload) != {"model", "messages"}:
        raise ValueError("Only model/messages are allowed; omit every decoding hyperparameter")
    opener, sleeper = opener or urllib.request.urlopen, sleeper or time.sleep
    attempts, started, backoff = [], time.perf_counter(), 0.0
    result = {"status": "error", "error_kind": "network_retry_exhausted", "response": None,
              "finish_reason": None, "usage": {}}
    for index in range(offset, 6):
        result.update(http_status=None, error=None)
        exception_class, reason_class = None, None
        attempt_started_at = now()
        if on_attempt:
            on_attempt({"number": index + 1, "started_at": attempt_started_at})
        tick, retry = time.perf_counter(), False
        request = urllib.request.Request(base_url + "/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode(), headers={"Content-Type": "application/json"})
        try:
            with opener(request, timeout=timeout) as stream:
                body = json.load(stream)
            choice = body["choices"][0]
            text = choice["message"].get("content")
            if choice.get("finish_reason") == "content_filter":
                result.update(status="error", error_kind="provider_guardrail", response=text, finish_reason="content_filter")
            elif not isinstance(text, str):
                raise ValueError("Model returned no text")
            else:
                result.update(status="ok", error_kind=None, error=None, response=text, http_status=200,
                    finish_reason=choice.get("finish_reason"), usage=body.get("usage") or {},
                    provider_response_id=body.get("id"), provider_model=body.get("model"))
        except urllib.error.HTTPError as error:
            exception_class = type(error).__name__
            detail = error.read().decode("utf-8", errors="replace")[:2000]
            lowered = detail.lower()
            guard = any(code in lowered for code in ("data_inspection_failed", "content_filter", "safety_violation"))
            retry = not guard and (error.code in (408, 409, 425, 429) or 500 <= error.code <= 599)
            result.update(status="error", response=None, error_kind="provider_guardrail" if guard else "network_error" if retry else "http_error",
                          error=detail, http_status=error.code)
        except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.IncompleteRead, OSError) as error:
            exception_class = type(error).__name__
            reason_class = type(error.reason).__name__ if isinstance(error, urllib.error.URLError) else None
            retry = True
            result.update(status="error", response=None, error_kind="network_error", error=f"{type(error).__name__}: {error}")
        except (ValueError, KeyError, IndexError, TypeError) as error:
            exception_class = type(error).__name__
            result.update(status="error", response=None, error_kind="format_error", error=f"{type(error).__name__}: {error}")
        attempts.append({"number": index + 1, "started_at": attempt_started_at, "finished_at": now(),
            "wall_seconds": time.perf_counter() - tick, "status": result["status"],
            "error_kind": result["error_kind"], "exception_class": exception_class,
            "reason_class": reason_class, "http_status": result.get("http_status"), "error": result.get("error")})
        if not retry or index == 5:
            break
        delay = min(30, 2 ** index)
        attempts[-1]["backoff_seconds"] = delay
        backoff += delay
        sleeper(delay)
    return {**result, "attempts": attempts, "wall_seconds": time.perf_counter() - started,
            "backoff_seconds": backoff, "request_sha256": digest(payload),
            "response_sha256": digest(result["response"]) if isinstance(result["response"], str) else None}


def execute(args):
    root = find_data_root(args.data_root)
    jobs, data = select_jobs(root, args.limit)
    print(json.dumps({"scope": "real_data_generation_smoke_no_judge", "data_root": str(root),
        "requests": len(jobs), "by_task": dict(Counter(j["record"]["benchmark"] + "/" + j["record"]["task"] for j in jobs)),
        "concurrency": 1, "generation_parameters_sent": []}, ensure_ascii=False, indent=2), flush=True)
    if args.plan_only:
        return 0
    server = read_json(args.server_dir / "server.json")
    if digest(server["settings"]) != server["signature"] or checkpoint_identity(server["model_path"]) != server["settings"]["model"]:
        raise ValueError("Server receipt or checkpoint identity changed")
    run = args.run_dir or REPO / "runs" / ("vllm-smoke-" + now().replace(":", "-") + "-" + uuid.uuid4().hex[:6])
    run = Path(run).resolve()
    signature = {"server_signature": server["signature"], "selection_sha256": digest(jobs),
        "data_manifest_sha256": sha_file(root / "prepared/manifest.json"),
        "script_sha256": sha_file(__file__), "prompt_code_sha256": sha_file(REPO / "src/mm_sft_lab/benchmarks.py"),
        "selection_code_sha256": sha_file(REPO / "src/mm_sft_lab/data.py"),
        "timeout_seconds": args.timeout_seconds, "network_retries": 5, "concurrency": 1}
    fingerprint = digest(signature)
    with stage_lock(run, "smoke"):
        manifest_path = run / "manifest.json"
        if manifest_path.exists():
            if not args.resume or read_json(manifest_path)["run_fingerprint"] != fingerprint:
                raise ValueError("Existing smoke differs; use identical settings with --resume or choose a new run directory")
        else:
            if args.resume:
                raise ValueError("Cannot resume without a manifest")
            if any(path.name != ".smoke.lock" for path in run.iterdir()):
                raise ValueError("A new smoke run directory must be empty")
            write_json(manifest_path, {"schema_version": 1, "created_at": now(), "status": "initialized",
                "scope": "generation_integrity_only_no_safety_or_capability_score", "is_mock": False,
                "run_fingerprint": fingerprint, "signature": signature, "server": server, "expected_requests": len(jobs)})
            write_json(run / "selection.json", jobs)
        previous = {}
        expected = {j["job_id"] for j in jobs}
        for row in read_jsonl(run / "predictions.jsonl", repair_tail=True):
            if row["job_id"] not in expected or row["job_id"] in previous or row["run_fingerprint"] != fingerprint:
                raise ValueError("Mixed or duplicate prediction identity")
            if row["status"] == "ok" and digest(row["response"]) != row["response_sha256"]:
                raise ValueError("Cached response changed")
            previous[row["job_id"]] = row
        offsets = Counter()
        for event in read_jsonl(run / "transport-attempts.jsonl", repair_tail=True):
            if event["run_fingerprint"] != fingerprint or event["job_id"] not in expected or event["number"] != offsets[event["job_id"]] + 1 or event["number"] > 6:
                raise ValueError("Unexpected transport retry history")
            offsets[event["job_id"]] = event["number"]
        remaining = [j for j in jobs if j["job_id"] not in previous]
        print("Smoke output:", run, flush=True)
        waited = wait_for_server(server, args.wait_seconds) if remaining else 0
        started = time.perf_counter()
        for job in remaining:
            payload = make_payload(job["record"], root, data, server["model_id"])
            def event(value):
                append_jsonl(run / "transport-attempts.jsonl", {**value, "job_id": job["job_id"], "run_fingerprint": fingerprint})
            result = completion(server["base_url"], payload, args.timeout_seconds, offsets[job["job_id"]], event)
            result.update(job_id=job["job_id"], id=job["record"]["id"], trial=job["trial"],
                          task=job["record"]["benchmark"] + "/" + job["record"]["task"],
                          recorded_at=now(), launch_id=server["launch_id"], run_fingerprint=fingerprint)
            append_jsonl(run / "predictions.jsonl", result)
            previous[job["job_id"]] = result
            print(f"[{len(previous)}/{len(jobs)}] {job['job_id']}: {result['status']}; finish={result['finish_reason']}; {result['wall_seconds']:.2f}s", flush=True)
            if result.get("http_status") in (400, 401, 403, 404) and result["error_kind"] != "provider_guardrail":
                break
        notes = Counter()
        by_task = defaultdict(list)
        for row in previous.values():
            by_task[row["task"]].append(row)
            if row.get("finish_reason") == "length":
                notes["truncated"] += 1
            if row["status"] == "ok" and not row["response"].strip():
                notes["empty"] += 1
            if row["status"] == "ok" and row.get("finish_reason") not in ("stop", "length"):
                notes["unexpected_finish_reason"] += 1
        successful = sum(r["status"] == "ok" for r in previous.values())
        complete = successful == len(jobs)
        summary = {"status": "passed_generation_only" if complete and not notes else "needs_review" if complete else "incomplete",
            "expected_requests": len(jobs), "successful_requests": successful, "error_requests": len(previous) - successful,
            "missing_requests": len(jobs) - len(previous), "issue_counts": dict(notes),
            "finish_reasons": dict(Counter(str(r.get("finish_reason")) for r in previous.values())),
            "scope": "Real generation only; no judge/API safety scores", "judge_calls": 0,
            "new_requests_this_session": len(previous) - (len(jobs) - len(remaining)),
            "server_readiness_wait_seconds": waited, "generation_session_wall_seconds": time.perf_counter() - started,
            "by_task": {task: {"requests": len(rows), "successful": sum(r["status"] == "ok" for r in rows),
                "sum_request_wall_seconds": sum(r["wall_seconds"] for r in rows),
                "output_tokens": sum((r.get("usage") or {}).get("completion_tokens", 0) or 0 for r in rows)} for task, rows in by_task.items()},
            "run_dir": str(run), "run_fingerprint": fingerprint}
        write_json(run / "summary.json", summary)
        manifest = read_json(manifest_path)
        write_json(manifest_path, {**manifest, **summary, "updated_at": now()})
        previews = []
        for job in jobs:
            row = previous.get(job["job_id"])
            if row:
                previews.extend([job["job_id"], "Question: " + render_text(job["record"]),
                                 "Response:\n" + (row.get("response") or row.get("error", "")), ""])
        (run / "answers.txt").write_text("\n".join(previews), encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return 0 if summary["status"] == "passed_generation_only" else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-dir", type=Path, default=DEFAULT_SERVER)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--limit", type=int, default=1, help="SIUO questions and complete MSS pairs per task; default means 7 requests")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--wait-seconds", type=int, default=900)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args(argv)
    if min(args.limit, args.wait_seconds, args.timeout_seconds) < 1:
        parser.error("limit/wait/timeout must be positive")
    return execute(args)


if __name__ == "__main__":
    raise SystemExit(main())
