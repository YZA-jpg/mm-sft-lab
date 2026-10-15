"""Separate CPU planning/auditing, GPU environment checks, training smoke and full SFT."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mm_sft_lab.training_data import load_train_config, token_audit, train_plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "preflight", "audit", "smoke", "train", "merge"):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
        if name not in ("preflight", "merge"):
            command.add_argument("--data-root", type=Path, default=Path("data/step1"))
        if name in ("audit", "smoke", "train", "merge"):
            command.add_argument("--output", type=Path, required=True)
        if name in ("audit", "smoke"):
            command.add_argument("--limit", type=int, default=64 if name == "smoke" else None)
        if name in ("smoke", "train"):
            command.add_argument("--audit-dir", type=Path, required=name == "train")
            command.add_argument("--resume-from", type=Path)
        if name == "smoke":
            command.add_argument("--steps", type=int, default=2)
        if name == "merge":
            command.add_argument("--adapter-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        config = load_train_config(args.config)
        if args.command == "plan":
            result = train_plan(args.data_root, config)
        elif args.command == "audit":
            result = token_audit(args.data_root, config, args.output, limit=args.limit)
        else:
            from mm_sft_lab.training import merge_adapter, run_training, training_preflight
            if args.command == "preflight":
                result = training_preflight(config)
            elif args.command == "merge":
                result = merge_adapter(config, args.adapter_dir, args.output)
            else:
                result = run_training(args.data_root, config, args.output, phase=args.command,
                    audit_dir=args.audit_dir, limit=getattr(args, "limit", 64), steps=getattr(args, "steps", 2), resume_from=args.resume_from)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["status"] != "failed" else 2
    except Exception as error:
        print(f"{args.command} failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
