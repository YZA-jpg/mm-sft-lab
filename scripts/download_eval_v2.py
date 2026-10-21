"""Download v2 sources sequentially; one failure does not block later sources."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation_v2.sources import download_all, HF_SOURCES, UPSTREAM


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=Path("data/eval-v2/raw"))
    parser.add_argument("--sources", nargs="+", default=["prompts", "figstep", "holisafe", "mmstar", "mathvista", "mmvet"],
                        choices=["prompts", "figstep", "holisafe", "mmstar", "mathvista", "mmvet", "wildguard-model"])
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if args.plan_only:
        print(json.dumps({"mode": "plan_no_network", "sources": args.sources, "hf": HF_SOURCES, "upstream": UPSTREAM}, indent=2))
        return 0
    result = download_all(args.sources, args.raw_root)
    return int(any(r["status"] != "downloaded" for r in result.values()))


if __name__ == "__main__":
    raise SystemExit(main())
