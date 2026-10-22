"""Generation-only concurrency sweep with one owned vLLM server and a fixed subset."""

import copy
import csv
import io
import random
import sys
import time
from pathlib import Path

from .common import atomic_text, digest, lock, now, write_json
from .config import generation_parameters
from .data import load_dataset
from .metrics import export_report
from .runner import initialize, stage
from .serving import Monitor, checkpoint_identity, runtime_info, serve


def sweep(data_root, output, config, model_path, *, levels=(1, 4, 8, 16), rounds=2,
          limit=4, gpu="7", vllm_python=sys.executable, adapter=None, memory=None, target_kv_cache_gib=None):
    if not levels or any(type(n) is not int or n < 1 for n in levels) or len(set(levels)) != len(levels):
        raise ValueError("Concurrency levels must be distinct positive integers")
    if type(rounds) is not int or rounds < 1 or type(limit) is not int or limit < 1:
        raise ValueError("Rounds/limit must be positive integers")
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("Sweep output already exists; use a new directory to avoid cached timing")
    base = copy.deepcopy(config)
    base["execution"]["per_benchmark_limit"] = limit
    if adapter:
        base["target"]["model"] = "target_adapter"
    port = int(base["target"]["base_url"].split(":")[-1].split("/")[0])
    if base["target"]["base_url"] != f"http://127.0.0.1:{port}/v1" or base["target"]["model"] != ("target_adapter" if adapter else "target"):
        raise ValueError("Managed sweep uses a loopback endpoint and target/target_adapter")
    rows, _ = load_dataset(data_root, base)
    selected_sha = digest(rows)
    identity = checkpoint_identity(model_path, adapter=adapter)
    identity["serving_runtime"] = {"python": str(vllm_python), "versions": runtime_info(vllm_python),
                                    "gpu": str(gpu), "memory_utilization": memory,
                                    "target_kv_cache_gib": target_kv_cache_gib, "target_limit_mm_per_prompt": {"image": 1, "video": 0}}
    output.mkdir(parents=True)
    schedule = []
    for repeat in range(rounds):
        order = list(levels)
        random.Random(20261021 + repeat).shuffle(order)  # scheduling only, never sent as a generation seed
        schedule.extend((repeat, level) for level in order)
    write_json(output / "plan.json", {"mode": "generation_only_no_cloud_judging", "created_at": now(),
        "selection_sha256": selected_sha, "selected_samples": len(rows), "schedule": schedule,
        "model_identity": identity, "http_generation_parameters": generation_parameters(config),
        "notes": ["One server startup shared by all measured rounds.", "A separate warmup run is excluded from measured results.",
                  "vLLM prefix/image caches may persist between levels; this measures warm steady-state behavior, not cold per-request performance.",
                  "Only the configured target output budget is sent; sampling controls and seed remain omitted, so responses can differ in length.",
                  "Do not infer linear multi-GPU scaling from this single-GPU sweep."]})
    summary, started = [], time.perf_counter()
    with lock(output, "sweep"), Monitor(output, gpu), serve(vllm_python, model_path, "target", port, gpu,
            output / "services/target", adapter=adapter, memory=memory,
            kv_cache_gib=target_kv_cache_gib, image_only=True):
        warmup = copy.deepcopy(base)
        warmup["execution"]["per_benchmark_limit"] = 1
        warmup["execution"]["generation_concurrency"] = min(levels)
        initialize(data_root, output / "warmup", warmup, identity)
        result = stage(output / "warmup", "generation")
        if not result["scoring_complete"]:
            raise ValueError("Warmup generation failed; inspect results before benchmarking")
        for repeat, level in schedule:
            run = output / f"round-{repeat + 1}-concurrency-{level}"
            current = copy.deepcopy(base)
            current["execution"]["generation_concurrency"] = level
            manifest = initialize(data_root, run, current, identity)
            if manifest["signature"]["selection_sha256"] != selected_sha:
                raise ValueError("Concurrency runs must use exactly the same inputs")
            with Monitor(run, gpu):
                result = stage(run, "generation")
            report = export_report(run)
            phase = report["timing"]["generation"]
            all_jobs = phase["observed_jobs"]
            wall = phase["wall_seconds_completed_sessions"]
            summary.append({"round": repeat + 1, "concurrency": level, "benchmark": "ALL",
                "stage_wall_seconds": wall, **all_jobs,
                "requests_per_second": all_jobs["successful_jobs"] / wall if wall else None,
                "output_tokens_per_second": all_jobs["completion_tokens"] / wall if wall else None,
                "gpu_mean_utilization_percent": report["resources"]["gpu"].get("utilization_percent", {}).get("mean"),
                "gpu_max_memory_used_MiB": report["resources"]["gpu"].get("memory_used_MiB", {}).get("max")})
            summary.extend({"round": repeat + 1, "concurrency": level, "benchmark": name, **values}
                           for name, values in phase["by_benchmark"].items())
            write_json(output / "sweep.json", {"selection_sha256": selected_sha, "rows": summary,
                                               "elapsed_seconds_so_far": time.perf_counter() - started})
            if not result["scoring_complete"]:
                raise ValueError("Measured generation failed; partial timings retained, do not select concurrency from failed trials")
    stream = io.StringIO(newline="")
    fields = ["round", "concurrency", "benchmark", "successful_jobs", "stage_wall_seconds", "span_seconds_excluding_restart_gaps",
              "requests_per_second", "output_tokens_per_second", "latency_p50_seconds", "latency_p95_seconds",
              "gpu_mean_utilization_percent", "gpu_max_memory_used_MiB"]
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(summary)
    atomic_text(output / "sweep.csv", stream.getvalue())
    return summary
