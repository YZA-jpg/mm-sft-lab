"""User-operated detached fresh smoke/full launcher; full requires an accepted matching smoke."""

import argparse
import copy
import json
import os
import re
import socket
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation_v2.common import atomic_text, digest, file_hash, latest, lock, read_journal, write_json
from evaluation_v2.config import load
from evaluation_v2.data import load_dataset
from evaluation_v2.prompts import request_preflight
from evaluation_v2.runner import generation_jobs, load_run
from evaluation_v2.serving import checkpoint_identity

REPO = Path(__file__).resolve().parents[1]


def launch_detached(command, output):
    with (output / "job.log").open("a", encoding="utf-8") as stream:
        return subprocess.Popen(command, cwd=REPO, stdin=subprocess.DEVNULL, stdout=stream,
                                stderr=subprocess.STDOUT, start_new_session=True)


def accept_smoke(run_dir, data_root, full_config, smoke_config, checkpoint=None):
    root = Path(run_dir).resolve()
    if any((root / name).exists() for name in (".pipeline.lock", "generation/.stage.lock")):
        raise ValueError("Smoke is still locked; wait for it to finish")
    manifest, records, data = load_run(root)
    if manifest["is_mock"]:
        raise ValueError("A mock smoke cannot authorize a real full run")
    if manifest["signature"]["config"] != smoke_config:
        raise ValueError("Smoke config differs from the current fixed smoke")
    expected_full = copy.deepcopy(smoke_config)
    expected_full["execution"]["sample_ids"] = None
    if expected_full != full_config:
        raise ValueError("Full configuration differs beyond smoke selection")
    if file_hash(Path(data_root) / "manifest.json") != manifest["signature"]["data_manifest_sha256"]:
        raise ValueError("Full dataset differs from smoke")
    identity = dict(manifest["signature"]["model_identity"])
    identity.pop("serving_runtime", None)
    if checkpoint is not None and identity != checkpoint:
        raise ValueError("Model/adapter differs from smoke; run its own smoke first")
    jobs = {j["job_id"]: j for j in generation_jobs(records, smoke_config)}
    rows = latest(read_journal(root / "generation/results.jsonl"))
    if set(rows) != set(jobs) or any(r["status"] != "ok" for r in rows.values()):
        raise ValueError("Smoke generation is incomplete or has errors")
    for row in rows.values():
        if (row.get("run_fingerprint") != manifest["run_fingerprint"] or
                digest(row["text"]) != row["response_sha256"] or row.get("finish_reason") not in ("stop", "length")):
            raise ValueError("Invalid smoke generation evidence")
    boundaries = read_journal(root / "generation/benchmark-sessions.jsonl")
    completed = [e for e in boundaries if e["event"] == "completed"]
    if any(e.get("run_fingerprint") != manifest["run_fingerprint"] for e in boundaries):
        raise ValueError("Smoke timing evidence belongs to another run")
    if [e["benchmark"] for e in completed] != smoke_config["benchmarks"]:
        raise ValueError("Smoke benchmark timing is incomplete; a fresh serial smoke is required")
    for left, right in zip(completed, completed[1:]):
        if left["finished_epoch"] > right["started_epoch"]:
            raise ValueError("Smoke benchmark timings overlap")
    checked = request_preflight(records, manifest["data_root"], data)
    if checked["unique_image_mime_counts"].get("image/webp") != 5:
        raise ValueError("Smoke did not cover the five real WebP images")
    return {"status": "accepted_generation_smoke_only", "run_dir": str(root),
            "run_fingerprint": manifest["run_fingerprint"], "successful_jobs": len(rows),
            "webp_images_checked": 5,
            "truncated_jobs": sum(r["finish_reason"] == "length" for r in rows.values()),
            "scope": "Generation/preparation/timing validation; no semantic safety validation"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("smoke", "full"))
    parser.add_argument("--gpu", default="1")
    parser.add_argument("--model-path", type=Path, default=Path("/data1/sharehub/models/Qwen3-VL-4B-Instruct"))
    parser.add_argument("--model-label", default="B0")
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("data/eval-v2/siuo-mss-stream-v1"))
    parser.add_argument("--vllm-python", default=".venv-vllm/bin/python")
    parser.add_argument("--smoke-run", type=Path)
    parser.add_argument("--check-only", action="store_true", help="Offline request plan only; no GPU/server/API")
    args = parser.parse_args(argv)
    os.chdir(REPO)
    config_path = Path(f"configs/eval_v2.siuo_mss_fixed_{args.phase}.json")
    config = load(config_path)
    if args.adapter_path:
        config["target"]["model"] = "target_adapter"
    records, data = load_dataset(args.data_root, config)
    checked = request_preflight(records, args.data_root, data)
    jobs = generation_jobs(records, config)
    planned = {"phase": args.phase, "generation_only": True, "judge_calls": 0,
               "sample_counts": dict(Counter(r["benchmark"] for r in records)),
               "generation_jobs": dict(Counter(j["benchmark"] for j in jobs)),
               "total_generation_jobs": len(jobs), "request_preflight": checked,
               "benchmark_schedule": "serial", "benchmark_order": config["benchmarks"],
               "generation_concurrency": 4, "target_parameters": config["generation"]}
    print(json.dumps(planned, ensure_ascii=False, indent=2), flush=True)
    if args.check_only:
        return 0
    if os.name != "posix" or not args.gpu.isdigit():
        raise ValueError("Actual launch requires Linux and one physical GPU index")
    checkpoint = checkpoint_identity(args.model_path, label=args.model_label, adapter=args.adapter_path)
    acceptance = None
    if args.phase == "full":
        if args.smoke_run is None:
            raise ValueError("Full launch requires --smoke-run; smoke never launches full automatically")
        smoke_config = load("configs/eval_v2.siuo_mss_fixed_smoke.json")
        if args.adapter_path:
            smoke_config["target"]["model"] = "target_adapter"
        acceptance = accept_smoke(args.smoke_run, args.data_root, config, smoke_config, checkpoint)
    # Resource checks happen only for a user-requested actual launch.
    free = subprocess.run(["nvidia-smi", "-i", args.gpu, "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                          check=True, capture_output=True, text=True, timeout=15)
    if int(free.stdout.strip()) < 55000:
        raise ValueError(f"GPU {args.gpu} has less than 55000 MiB free; choose another GPU before launching")
    try:
        with socket.create_connection(("127.0.0.1", 18014), timeout=1):
            raise ValueError("Port 18014 is occupied; inspect its owner before launching")
    except (ConnectionRefusedError, socket.timeout):
        pass
    label = re.sub(r"[^A-Za-z0-9_-]", "_", args.model_label)
    if not label:
        raise ValueError("Model label must contain a letter or number")
    runs = REPO / "runs"
    runs.mkdir(exist_ok=True)
    pointer = runs / f"latest-{label}-v2-fixed-{args.phase}.txt"
    with lock(runs, f"launch-fixed-{label}-{args.phase}"):
        if pointer.exists():
            prior = Path(pointer.read_text().strip())
            pidfile = prior / "job.pid"
            if pidfile.exists():
                try:
                    os.kill(int(pidfile.read_text().strip()), 0)
                except ProcessLookupError:
                    pass
                else:
                    raise ValueError("Previous phase PID still exists; inspect it before launching again")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        output = runs / f"{label}-v2-fixed-{args.phase}-{stamp}"
        output.mkdir(exist_ok=False)
        write_json(output / "launch-plan.json", planned)
        if acceptance:
            write_json(output / "accepted-smoke.json", acceptance)
        command = ["bash", "scripts/run_eval_v2.sh", "--config", str(config_path), "--data-root", str(args.data_root),
                   "--model-path", str(args.model_path), "--model-label", args.model_label, "--run-dir", str(output),
                   "--gpu", args.gpu, "--vllm-python", args.vllm_python,
                   "--gpu-memory-utilization", "0.65", "--target-kv-cache-gib", "40", "--generate-only"]
        if args.adapter_path:
            command += ["--adapter-path", str(args.adapter_path)]
        git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        atomic_text(output / "git-commit.txt", git.stdout)
        write_json(output / "launch-command.json", {"argv": command})
        process = launch_detached(command, output)
        atomic_text(output / "job.pid", str(process.pid) + "\n")
        atomic_text(pointer, str(output) + "\n")
    print(json.dumps({"status": "detached_generation_started", "pid": process.pid,
                      "run_dir": str(output), "pointer": str(pointer), "log": str(output / "job.log"),
                      "full_started_automatically": False}, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
