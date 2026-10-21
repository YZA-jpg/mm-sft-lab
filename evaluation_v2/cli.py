"""Explicit plan, environment, generation, judge and report commands."""

import argparse
import importlib.metadata
import json
import os
import platform
import signal
import subprocess
import sys
from collections import Counter
from pathlib import Path

from . import config as configuration
from .common import read_json
from .data import load_dataset
from .metrics import export_report
from .reporting import compare
from .status import status
from .runner import generation_jobs, initialize, stage
from .serving import checkpoint_identity, run_all, runtime_info, server_command


def parser():
    root = argparse.ArgumentParser(description="Independent v2 evaluation; all HTTP decoding hyperparameters omitted")
    commands = root.add_subparsers(dest="command", required=True)
    for command in ("plan", "init", "run"):
        sub = commands.add_parser(command)
        sub.add_argument("--config", type=Path, default=Path("configs/eval_v2.smoke.json"))
        sub.add_argument("--data-root", type=Path, required=True)
        sub.add_argument("--model-path", type=Path)
        sub.add_argument("--model-label")
        sub.add_argument("--adapter-path", type=Path)
        sub.add_argument("--mock", action="store_true", help="Synthetic data/model identity only")
        sub.add_argument("--resume", action="store_true")
        if command != "plan":
            sub.add_argument("--run-dir", type=Path, required=True)
        if command in ("plan", "run"):
            sub.add_argument("--wildguard-path", type=Path)
            sub.add_argument("--vllm-python", default=sys.executable)
            sub.add_argument("--gpu", default="7")
            sub.add_argument("--gpu-memory-utilization", type=float)
            sub.add_argument("--target-kv-cache-gib", type=float)
            sub.add_argument("--wildguard-kv-cache-gib", type=float)
        if command == "run":
            sub.add_argument("--generate-only", action="store_true", help="Run target generation and release GPU; no judge API calls")
    for command in ("generate", "judge-deepseek", "judge-wildguard", "score", "status"):
        sub = commands.add_parser(command)
        sub.add_argument("--run-dir", type=Path, required=True)
        if command not in ("score", "status"):
            sub.add_argument("--retry-recoverable", action="store_true", help="After fixing authentication/local preparation only; never rejudge guards/refusals/format errors")
    sub = commands.add_parser("preflight")
    sub.add_argument("--model-path", type=Path)
    sub.add_argument("--adapter-path", type=Path)
    sub.add_argument("--vllm-python", default=sys.executable)
    sub = commands.add_parser("compare")
    sub.add_argument("--reports", nargs="+", type=Path, required=True)
    sub.add_argument("--output", type=Path, required=True)
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    if os.name == "posix":
        signal.signal(signal.SIGTERM, interrupted)
    if args.command == "status":
        print(json.dumps(status(args.run_dir), ensure_ascii=False, indent=2))
        return 0
    if args.command == "compare":
        compare(args.reports, args.output)
        print(args.output / "main-table.md")
        return 0
    if args.command == "preflight":
        result = {"python": sys.version, "platform": platform.platform(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                  "scope": "Read-only package/checkpoint/GPU query; no weight loading, server start or API requests"}
        try:
            result["vllm_environment"] = runtime_info(args.vllm_python)
        except Exception as error:
            result["vllm_environment"] = {"error": str(error)}
        if args.model_path:
            try:
                result["checkpoint"] = checkpoint_identity(args.model_path, adapter=args.adapter_path)
            except Exception as error:
                result["checkpoint"] = {"error": str(error)}
        try:
            gpu = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free,driver_version", "--format=csv"], capture_output=True, text=True, timeout=10)
            result["gpu"] = gpu.stdout.strip() or gpu.stderr.strip()
        except (OSError, subprocess.TimeoutExpired) as error:
            result["gpu"] = str(error)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command in ("generate", "judge-deepseek", "judge-wildguard"):
        phase = {"generate": "generation", "judge-deepseek": "deepseek", "judge-wildguard": "wildguard"}[args.command]
        result = stage(args.run_dir, phase, retry_recoverable=args.retry_recoverable)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["scoring_complete"] else 2
    if args.command == "score":
        result = export_report(args.run_dir)
        print((args.run_dir / "report/table.md").read_text(encoding="utf-8"))
        return 0 if result["complete"] else 2
    config = configuration.load(args.config)
    if args.adapter_path:
        config["target"]["model"] = "target_adapter"
    if args.command == "plan":
        rows, data = load_dataset(args.data_root, config)
        jobs = generation_jobs(rows, config)
        counts = Counter(job["benchmark"] for job in jobs)
        result = {"mode": "plan_no_API_no_GPU", "protocol_id": config["protocol_id"], "protocol_status": config["protocol_status"],
                  "is_mock": data["is_mock"], "sample_counts": dict(Counter(r["benchmark"] for r in rows)),
                  "generation_jobs": dict(counts), "total_generation_jobs": len(jobs),
                  "deepseek_jobs": sum(n * (config["mmvet_judge_repeats"] if name == "mmvet" else 1)
                                       for name, n in counts.items() if name not in ("wildjailbreak", "mmstar")),
                  "wildguard_jobs": counts["wildjailbreak"], "http_decoding_parameters": [],
                  "network_retries": 5, "max_total_transport_attempts_per_job": 6,
                  "concurrency": config["execution"], "asr_group_rule": "all k valid; any unsafe",
                  "managed_gpu_schedule": "target generation first; stop target; run WildGuard and cloud DeepSeek concurrently"}
        if args.model_path:
            port = int(config["target"]["base_url"].split(":")[-1].split("/")[0])
            result["planned_target_server_argv"] = server_command(args.vllm_python, args.model_path, "target", port,
                            adapter=args.adapter_path, memory=args.gpu_memory_utilization,
                            kv_cache_gib=args.target_kv_cache_gib, image_only=True)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "init":
        if args.mock:
            identity = {"kind": "mock", "label": args.model_label or "mock-target"}
        elif args.model_path:
            identity = checkpoint_identity(args.model_path, label=args.model_label, adapter=args.adapter_path)
        else:
            raise ValueError("Supply --model-path for the real model identity or --mock for synthetic validation")
        result = initialize(args.data_root, args.run_dir, config, identity, resume=args.resume)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    if args.mock or not args.model_path:
        raise ValueError("Managed real run requires a local --model-path; use demo for synthetic execution")
    if not args.generate_only and any(b not in ("wildjailbreak", "mmstar") for b in config["benchmarks"]):
        key_env = config["deepseek"]["api_key_env"]
        if key_env and not os.environ.get(key_env, "").strip():
            raise ValueError(f"Set {key_env} before starting the GPU phases")
    result = run_all(args.data_root, args.run_dir, config, args.model_path, args.wildguard_path,
                    label=args.model_label, adapter=args.adapter_path, gpu=args.gpu,
                    vllm_python=args.vllm_python, memory=args.gpu_memory_utilization,
                    target_kv_cache_gib=args.target_kv_cache_gib, wildguard_kv_cache_gib=args.wildguard_kv_cache_gib,
                    generate_only=args.generate_only, resume=args.resume)
    if args.generate_only:
        summary = result["generation"]
        print(json.dumps({"scope": "generation_only_no_judge_calls", **summary,
                          "results": str(args.run_dir / "generation/results.jsonl"),
                          "semantic_scoring_complete": result["complete"]}, ensure_ascii=False, indent=2))
        return 0 if summary["successful"] == summary["expected"] else 2
    print((args.run_dir / "report/table.md").read_text(encoding="utf-8"))
    return 0 if result["complete"] else 2
