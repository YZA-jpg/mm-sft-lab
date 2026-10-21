"""Compare concurrency on one already-running vLLM service; no judge or decoding overrides."""

import argparse
import csv
import io
import json
import math
import statistics
import subprocess
import time
import uuid
from collections import Counter, defaultdict, deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from threading import Lock

from smoke_vllm_benchmarks import (TASKS, completion, find_data_root, make_payload,
                                   select_jobs, wait_for_server)
from start_vllm_model import (DEFAULT_SERVER, REPO, checkpoint_identity, digest,
                              now, read_json, sha_file, write_json)
from mm_sft_lab.common import append_jsonl, atomic_text, stage_lock


def fair_order(jobs):
    """Keep one fixed task-round-robin order at every concurrency level."""
    buckets = {task: deque() for task in TASKS}
    for job in jobs:
        row = job["record"]
        buckets[row["benchmark"] + "/" + row["task"]].append(job)
    ordered = []
    while any(buckets.values()):
        for bucket in buckets.values():
            if bucket:
                ordered.append(bucket.popleft())
    return ordered


def gpu_snapshot(gpu):
    """Whole-device readings include other users; snapshots are not peak memory."""
    try:
        if not str(gpu).isdigit():
            raise ValueError("Expected one physical GPU index")
        result = subprocess.run(["nvidia-smi", "-i", str(gpu),
            "--query-gpu=index,utilization.gpu,memory.used,memory.free",
            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
        if result.returncode:
            raise ValueError(result.stderr.strip()[:500])
        row, = list(csv.reader(io.StringIO(result.stdout)))
        index, utilization, used, free = [int(value.strip()) for value in row]
        return {"recorded_at": now(), "gpu": index, "utilization_percent": utilization,
                "used_MiB": used, "free_MiB": free, "scope": "whole shared device; not a peak reading"}
    except (OSError, ValueError, subprocess.TimeoutExpired) as error:
        return {"recorded_at": now(), "error": f"{type(error).__name__}: {error}"}


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower, upper = math.floor(index), math.ceil(index)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def summarize(rows, expected, wall):
    good = [row for row in rows if row["status"] == "ok"]
    issues = Counter()
    for row in good:
        if not row["response"].strip():
            issues["empty"] += 1
        if row.get("finish_reason") == "length":
            issues["truncated"] += 1
        elif row.get("finish_reason") != "stop":
            issues["unexpected_finish_reason"] += 1
    usages = [(row.get("usage") or {}).get("completion_tokens") for row in good]
    known = [value for value in usages if type(value) is int and value >= 0]
    tokens = sum(known) if len(known) == len(usages) and good else None
    tasks = defaultdict(list)
    for row in rows:
        tasks[row["task"]].append(row)
    complete = len(good) == expected
    return {"status": "passed_generation_only" if complete and not issues else "needs_review" if complete else "incomplete",
        "expected_requests": expected, "successful_requests": len(good),
        "error_requests": len(rows) - len(good), "missing_requests": expected - len(rows),
        "wall_seconds": wall, "successful_requests_per_second": len(good) / wall if wall > 0 else None,
        "latency_p50_seconds": percentile([r["wall_seconds"] for r in rows], .5),
        "latency_p95_seconds": percentile([r["wall_seconds"] for r in rows], .95),
        "output_tokens": tokens, "requests_with_output_token_usage": len(known),
        "output_tokens_per_second": tokens / wall if tokens is not None and wall > 0 else None,
        "issue_counts": dict(issues), "finish_reasons": dict(Counter(str(r.get("finish_reason")) for r in rows)),
        "performance_comparable": complete and not issues,
        "by_task": {task: {"requests": len(values), "successful": sum(r["status"] == "ok" for r in values),
            "sum_request_latency_seconds": sum(r["wall_seconds"] for r in values),
            "active_span_seconds": max(r["finished_offset_seconds"] for r in values) - min(r["started_offset_seconds"] for r in values),
            "output_tokens_observed": sum(value for r in values if r["status"] == "ok"
                for value in [(r.get("usage") or {}).get("completion_tokens")] if type(value) is int and value >= 0)}
            for task, values in tasks.items()},
        "timing_scope": "Client HTTP phase including server queue, network retries, backoff and result writes; excludes payload preparation, readiness, warmup and GPU snapshots. Task spans overlap and must not be summed."}


def run_trial(jobs, payloads, server, folder, concurrency, timeout, fingerprint):
    folder.mkdir(parents=True, exist_ok=False)
    rows, writing = [], Lock()
    pending_jobs = deque(jobs)
    before = gpu_snapshot(server["gpu"])
    started_at, started = now(), time.perf_counter()
    write_json(folder / "summary.json", {"status": "running", "expected_requests": len(jobs),
        "concurrency": concurrency, "run_fingerprint": fingerprint, "started_at": started_at})

    def generate(job):
        begin = time.perf_counter() - started
        def event(value):
            with writing:
                append_jsonl(folder / "transport-attempts.jsonl", {**value,
                    "job_id": job["job_id"], "run_fingerprint": fingerprint})
        result = completion(server["base_url"], payloads[job["job_id"]], timeout, on_attempt=event)
        return {**result, "job_id": job["job_id"], "id": job["record"]["id"], "trial": job["trial"],
            "task": job["record"]["benchmark"] + "/" + job["record"]["task"],
            "recorded_at": now(), "launch_id": server["launch_id"], "run_fingerprint": fingerprint,
            "started_offset_seconds": begin, "finished_offset_seconds": time.perf_counter() - started}

    fatal = False
    try:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            running = {}
            def fill():
                while pending_jobs and not fatal and len(running) < concurrency:
                    job = pending_jobs.popleft()
                    running[pool.submit(generate, job)] = job
            fill()
            while running:
                done, _ = wait(running, return_when=FIRST_COMPLETED)
                for future in done:
                    running.pop(future)
                    row = future.result()
                    append_jsonl(folder / "predictions.jsonl", row)
                    rows.append(row)
                    fatal |= row.get("http_status") in (400, 401, 403, 404) and row.get("error_kind") != "provider_guardrail"
                    print(f"{folder.name} [{len(rows)}/{len(jobs)}] {row['job_id']}: {row['status']}; finish={row.get('finish_reason')}; {row['wall_seconds']:.2f}s", flush=True)
                fill()
        summary = summarize(rows, len(jobs), time.perf_counter() - started)
    except BaseException:
        summary = {**summarize(rows, len(jobs), time.perf_counter() - started),
                   "status": "interrupted", "performance_comparable": False}
        write_json(folder / "summary.json", {**summary, "concurrency": concurrency,
            "run_fingerprint": fingerprint, "started_at": started_at, "updated_at": now(), "gpu_before": before})
        raise
    summary.update(concurrency=concurrency, run_fingerprint=fingerprint,
                   started_at=started_at, updated_at=now(), gpu_before=before, gpu_after=gpu_snapshot(server["gpu"]))
    write_json(folder / "summary.json", summary)
    return summary


def comparison(trials, levels, rounds):
    results = []
    for level in levels:
        values = [row for row in trials if row["concurrency"] == level]
        valid = [row for row in values if row["performance_comparable"]]
        complete = len(valid) == rounds
        rates = [row["successful_requests_per_second"] for row in valid]
        token_rates = [row["output_tokens_per_second"] for row in valid if row["output_tokens_per_second"] is not None]
        wall = sum(row["wall_seconds"] for row in values)
        successes = sum(row["successful_requests"] for row in values)
        tokens_known = complete and all(row["output_tokens"] is not None for row in valid)
        tokens = sum(row["output_tokens"] for row in valid) if tokens_known else None
        results.append({"concurrency": level, "completed_rounds": len(values), "valid_rounds": len(valid),
            "requested_rounds": rounds, "performance_comparable": complete,
            "successful_requests": successes,
            "expected_requests": sum(r["expected_requests"] for r in values),
            "total_wall_seconds": wall,
            "pooled_requests_per_second": successes / wall if complete and wall > 0 else None,
            "total_output_tokens": tokens,
            "pooled_output_tokens_per_second": tokens / wall if tokens is not None and wall > 0 else None,
            "median_wall_seconds": statistics.median(r["wall_seconds"] for r in valid) if complete else None,
            "median_requests_per_second": statistics.median(rates) if complete else None,
            "median_output_tokens_per_second": statistics.median(token_rates) if complete and len(token_rates) == rounds else None,
            "truncated_responses": sum(r["issue_counts"].get("truncated", 0) for r in values)})
    baseline = next((r["pooled_requests_per_second"] for r in results if r["concurrency"] == 1), None)
    for row in results:
        rate = row["pooled_requests_per_second"]
        row["request_throughput_relative_to_c1"] = rate / baseline if baseline and rate is not None else None
    return results


def write_csv(path, rows):
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(path, output.getvalue())


def execute(args):
    root = find_data_root(args.data_root)
    jobs, data = select_jobs(root, args.limit)
    jobs = fair_order(jobs)
    server = read_json(args.server_dir / "server.json")
    if digest(server["settings"]) != server["signature"] or checkpoint_identity(server["model_path"]) != server["settings"]["model"]:
        raise ValueError("Server receipt or checkpoint identity changed")
    payloads = {j["job_id"]: make_payload(j["record"], root, data, server["model_id"]) for j in jobs}
    signature = {"server_signature": server["signature"], "selection_sha256": digest(jobs),
        "request_sha256": {key: digest(value) for key, value in payloads.items()},
        "data_manifest_sha256": sha_file(root / "prepared/manifest.json"),
        "code_sha256": {str(path.relative_to(REPO)): sha_file(path) for path in
            (Path(__file__), REPO / "scripts/smoke_vllm_benchmarks.py", REPO / "scripts/start_vllm_model.py",
             REPO / "src/mm_sft_lab/benchmarks.py", REPO / "src/mm_sft_lab/data.py")},
        "levels": args.levels, "rounds": args.rounds, "limit": args.limit, "warmup": args.warmup,
        "timeout_seconds": args.timeout_seconds, "network_retries": 5, "generation_parameters_sent": []}
    fingerprint = digest(signature)
    output = (args.output or REPO / "runs" / ("vllm-concurrency-" + now().replace(":", "-") + "-" + uuid.uuid4().hex[:6])).resolve()
    with stage_lock(output, "concurrency"):
        if any(p.name != ".concurrency.lock" for p in output.iterdir()):
            raise ValueError("Profiling requires a fresh output directory; existing runs are never overwritten or combined")
        manifest = {"schema_version": 1, "status": "initialized", "created_at": now(),
            "run_fingerprint": fingerprint, "signature": signature, "server": server,
            "requests_per_trial": len(jobs), "expected_measured_requests": len(jobs) * len(args.levels) * args.rounds,
            "warmup_requests": args.warmup, "judge_calls": 0,
            "scope": "Generation performance only; no safety/capability score; all response text is retained"}
        write_json(output / "manifest.json", manifest)
        write_json(output / "selection.json", jobs)
        print(json.dumps({"output": str(output), "gpu": server["gpu"], "base_url": server["base_url"],
            "requests_per_trial": len(jobs), "measured_requests": manifest["expected_measured_requests"],
            "warmup_requests": args.warmup, "levels": args.levels, "rounds": args.rounds,
            "generation_parameters_sent": []}, ensure_ascii=False, indent=2), flush=True)
        trials = []
        def save(status, **extra):
            manifest.update(status=status, updated_at=now(), **extra)
            write_json(output / "manifest.json", manifest)
            rows = comparison(trials, args.levels, args.rounds)
            write_csv(output / "comparison.csv", rows)
            write_json(output / "summary.json", {"status": status, "run_fingerprint": fingerprint,
                "judge_calls": 0, "server_readiness_wait_seconds": manifest.get("server_readiness_wait_seconds"),
                "trials": trials, "comparison": rows,
                "notes": ["Same image bytes/prompts at every level; responses and lengths can vary under server defaults.",
                    "Primary throughput is total successful requests (or output tokens) divided by summed measured wall time; per-round rate medians are descriptive only.",
                    "Warmup and readiness are excluded from measured trial time. GPU snapshots cover the whole shared card.",
                    "This small shared-GPU experiment does not establish linear multi-GPU speedup.",
                    "Interrupted runs retain outputs but must not be combined into a timing trial."]})
        try:
            ready = wait_for_server(server, args.wait_seconds)
            save("running", server_readiness_wait_seconds=ready)
            if args.warmup:
                warmed = [{**jobs[i % len(jobs)], "job_id": f"warmup-{i + 1}"} for i in range(args.warmup)]
                warm_payloads = {row["job_id"]: payloads[jobs[i % len(jobs)]["job_id"]] for i, row in enumerate(warmed)}
                warm = run_trial(warmed, warm_payloads, server, output / "warmup", 1, args.timeout_seconds, fingerprint)
                if warm["status"] != "passed_generation_only":
                    save("warmup_failed")
                    return 2
            for round_index in range(args.rounds):
                levels = args.levels if round_index % 2 == 0 else list(reversed(args.levels))
                for level in levels:
                    name = f"round-{round_index + 1:02d}-concurrency-{level}"
                    trial = run_trial(jobs, payloads, server, output / name, level, args.timeout_seconds, fingerprint)
                    trials.append({"round": round_index + 1, "directory": name, **trial})
                    save("running")
                    if trial["status"] == "incomplete":
                        save("stopped_on_request_errors")
                        return 2
            status = "complete" if all(t["performance_comparable"] for t in trials) else "needs_review"
            save(status)
            print((output / "comparison.csv").read_text(), flush=True)
            print("Comparison:", output / "comparison.csv", flush=True)
            return 0 if status == "complete" else 2
        except BaseException:
            save("interrupted")
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-dir", type=Path, default=DEFAULT_SERVER)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--levels", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--wait-seconds", type=int, default=900)
    args = parser.parse_args(argv)
    if min(args.levels + [args.rounds, args.limit, args.timeout_seconds, args.wait_seconds]) < 1 or args.warmup < 0 or len(set(args.levels)) != len(args.levels):
        parser.error("levels must be unique positive integers; rounds/limit/timeout/wait positive; warmup nonnegative")
    return execute(args)


if __name__ == "__main__":
    raise SystemExit(main())
