"""Stage-specific command line entry points."""

import argparse
import json
import sys
from pathlib import Path

from .common import digest
from .config import load_config, load_judge_config


def main(argv=None):
    parser = argparse.ArgumentParser(prog="mm-sft-lab", description="Multimodal baseline experiment stages")
    commands = parser.add_subparsers(dest="command", required=True)
    preflight = commands.add_parser("preflight", help="Inspect checkpoint and visible GPU, or API settings")
    preflight.add_argument("--config", type=Path, required=True)
    plan = commands.add_parser("plan", help="Validate data and print the selected request counts without loading a model")
    plan.add_argument("--config", type=Path, required=True)
    plan.add_argument("--data-root", type=Path, default=Path("data"))
    infer = commands.add_parser("generate", help="Generate or resume model responses")
    infer.add_argument("--config", type=Path, required=True)
    infer.add_argument("--data-root", type=Path, default=Path("data"))
    infer.add_argument("--run-dir", type=Path, required=True)
    infer.add_argument("--resume", action="store_true")
    infer.add_argument("--retry-errors", action="store_true")
    validate = commands.add_parser("validate-run", help="Check generation completeness, visual inputs and batch telemetry without judging")
    validate.add_argument("--run-dir", type=Path, required=True)
    batches = commands.add_parser("compare-batches", help="Compare the same greedy requests with only batch_size changed")
    batches.add_argument("--before", type=Path, required=True)
    batches.add_argument("--after", type=Path, required=True)
    batches.add_argument("--output", type=Path, required=True)
    judge = commands.add_parser("judge", help="Judge generated responses through a separate API")
    judge.add_argument("--config", type=Path, default=Path("configs/judge.example.json"))
    judge.add_argument("--run-dir", type=Path, required=True)
    judge.add_argument("--judge-dir", type=Path)
    judge.add_argument("--data-root", type=Path)
    judge.add_argument("--resume", action="store_true")
    judge.add_argument("--retry-errors", action="store_true")
    judge.add_argument("--mock", action="store_true", help="Synthetic labels for pipeline testing only")
    score = commands.add_parser("score", help="Write coverage-aware metrics, CSV, and Markdown report")
    score.add_argument("--run-dir", type=Path, required=True)
    score.add_argument("--judge-dir", type=Path)
    review = commands.add_parser("export-review", help="Export semantic tasks for human labeling; no API calls")
    review.add_argument("--run-dir", type=Path, required=True)
    review.add_argument("--output", type=Path, required=True)
    manual = commands.add_parser("import-judgments", help="Import human labels tied to exact responses")
    manual.add_argument("--run-dir", type=Path, required=True)
    manual.add_argument("--input", type=Path, required=True)
    manual.add_argument("--judge-dir", type=Path, required=True)
    manual.add_argument("--reviewer", default="human")
    manual.add_argument("--protocol", default="manual_lab_rubric_v1")
    submission = commands.add_parser("export-textvqa-test", help="Write predictions for official Test scoring")
    submission.add_argument("--run-dir", type=Path, required=True)
    submission.add_argument("--output", type=Path, required=True)
    compare = commands.add_parser("compare", help="Compare two complete reports under the same evaluation protocol")
    compare.add_argument("--before", type=Path, required=True)
    compare.add_argument("--after", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command in ("preflight", "plan", "generate"):
            config = load_config(args.config)
        if args.command == "preflight":
            from .backends import ChatClient
            from .preflight import environment, inspect_local, inference_identity
            result = (inspect_local(config["model_path"]) if config["backend"] == "qwen3_vl_hf" else
                      {"api": ChatClient(config["api"]).identity, "environment": environment()} if config["backend"] == "chat_api" else
                      {"backend": "mock", "is_mock": True, "environment": environment()})
            if config.get("adapter_path"):
                result["inference_target"] = inference_identity(config)
        elif args.command == "plan":
            from collections import Counter
            from .data import load_prepared
            records, _, manifest = load_prepared(args.data_root, config)
            result = {"backend": config["backend"], "provenance": manifest["provenance_status"],
                      "selected_requests": dict(Counter(f"{row['benchmark']}/{row['task']}" for row in records)),
                      "responses_per_request": len(config["seeds"]), "selection_sha256": digest(records),
                      "mssbench_pairs": len({row["pair_id"] for row in records if row["pair_id"]})}
        elif args.command == "generate":
            from .runner import generate
            if args.retry_errors and not args.resume:
                raise ValueError("--retry-errors requires --resume")
            result = generate(config, args.data_root, args.run_dir, resume=args.resume, retry_errors=args.retry_errors)
            print(json.dumps({key: result[key] for key in ("status", "expected_requests", "successful_requests", "error_requests", "missing_requests")}, indent=2))
            return 0 if result["status"] == "complete" else 2
        elif args.command in ("validate-run", "compare-batches"):
            from .diagnostics import compare_batches, validate_run
            result = (validate_run(args.run_dir) if args.command == "validate-run" else
                      compare_batches(args.before, args.after, args.output))
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return 0 if result["status"] == "passed" else 2
        elif args.command == "judge":
            from .judge import judge_run
            if args.retry_errors and not args.resume:
                raise ValueError("--retry-errors requires --resume")
            result = judge_run(args.run_dir, args.judge_dir or args.run_dir / "judge", load_judge_config(args.config),
                               data_root=args.data_root, resume=args.resume, retry_errors=args.retry_errors, mock=args.mock)
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return 0 if result["status"] == "complete" else 2
        elif args.command == "score":
            from .metrics import score_run
            result, output = score_run(args.run_dir, args.judge_dir)
            print(json.dumps({"complete": result["complete"], "is_mock": result["is_mock"], "output": str(output)}, indent=2))
            return 0
        elif args.command in ("export-review", "import-judgments", "export-textvqa-test"):
            from .manual_judge import export_review, import_manual, export_textvqa_test
            if args.command == "export-review":
                result = export_review(args.run_dir, args.output)
            elif args.command == "import-judgments":
                result = import_manual(args.run_dir, args.input, args.judge_dir, reviewer=args.reviewer, protocol=args.protocol)
            else:
                result = export_textvqa_test(args.run_dir, args.output)
        else:
            from .metrics import compare_runs
            result = compare_runs(args.before, args.after, args.output)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    except Exception as error:
        print(f"{args.command} failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
