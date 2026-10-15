"""Prepare a separate VLGuard training root; keep the validated evaluation snapshot."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mm_sft_lab.vlguard import prepare_vlguard


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/step1"), help="Locked downloads and prepared evaluation root")
    parser.add_argument("--output-root", type=Path, help="Defaults to <data-root>/vlguard")
    parser.add_argument("--benchmark-data-root", type=Path, help="Defaults to --data-root")
    args = parser.parse_args()
    try:
        result = prepare_vlguard(args.data_root, args.output_root or args.data_root / "vlguard",
                                 args.benchmark_data_root or args.data_root)
    except Exception as error:
        print(f"VLGuard preparation failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"status": result["status"], "is_mock": result["is_mock"], **result["audit"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
