"""Audit real prepared data read-only, writing a separate report under runs/."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mm_sft_lab.common import write_json
from mm_sft_lab.data_audit import audit_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/step1"))
    parser.add_argument("--vlguard-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.output.exists():
            raise ValueError("Audit report exists; select a new output file")
        for root in (args.data_root, args.vlguard_root):
            if root is not None and args.output.resolve().is_relative_to(root.resolve()):
                raise ValueError("Write the audit report outside frozen data roots")
        result = audit_data(args.data_root, Path(__file__).resolve().parents[1] / "configs/train", vlguard_root=args.vlguard_root)
        write_json(args.output, result)
    except Exception as error:
        print(f"Data audit failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 1 if result["violations"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
