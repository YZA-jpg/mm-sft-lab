"""Prepare TextVQA, HH and WildJailbreak; VLGuard conversion is not yet enabled."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mm_sft_lab.step1_data import prepare_step1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/step1"))
    parser.add_argument("--benchmark-data-root", type=Path, required=True, help="Existing locked SIUO/MSSBench root")
    parser.add_argument("--allow-unlocked", action="store_true", help="Synthetic fixtures only; output cannot be used for real training")
    args = parser.parse_args()
    try:
        result = prepare_step1(args.data_root, args.benchmark_data_root, allow_unlocked=args.allow_unlocked)
    except Exception as error:
        print(f"Preparation failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"status": result["status"], "is_mock": result["is_mock"],
                      "raw_training": result["audit"]["raw_training_counts"],
                      "training": result["audit"]["training_counts"], "excluded": result["audit"]["excluded_counts"],
                      "hh_format_excluded": result["audit"]["hh_format_excluded_count"],
                      "hh_test_context_issues": result["audit"]["hh_test_context_issue_count"],
                      "evaluation": result["request_counts"]}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
