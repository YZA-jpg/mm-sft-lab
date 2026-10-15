"""Download one source or all four sequentially, isolating per-source failures."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mm_sft_lab.step1_sources import SOURCE_NAMES, download_source
from mm_sft_lab.common import now, write_json


def download_all(data_root, *, max_workers=4):
    summary_path = Path(data_root).resolve() / "download_batch_summary.json"
    summary = {"schema_version": 1, "started_at": now(), "status": "running", "order": list(SOURCE_NAMES), "datasets": []}
    write_json(summary_path, summary)
    for index, name in enumerate(SOURCE_NAMES, start=1):
        print(f"\n[{index}/{len(SOURCE_NAMES)}] Starting {name}", flush=True)
        try:
            result = download_source(data_root, name, max_workers=max_workers)
        except Exception as error:
            message = f"{type(error).__name__}: {error}"
            summary["datasets"].append({"dataset": name, "status": "failed", "error": message})
            print(f"{name} failed; continuing to the next dataset. {message}", file=sys.stderr, flush=True)
        except KeyboardInterrupt:
            summary.update(status="interrupted", updated_at=now())
            write_json(summary_path, summary)
            raise
        else:
            summary["datasets"].append({"dataset": name, "status": "downloaded", "files": len(result["files"])})
            print(f"{name}: downloaded and verified", flush=True)
        summary["updated_at"] = now()
        write_json(summary_path, summary)
    summary.update(status="complete" if all(row["status"] == "downloaded" for row in summary["datasets"]) else "partial_failure",
                   finished_at=now())
    write_json(summary_path, summary)
    return {**summary, "summary_file": str(summary_path)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--dataset", choices=SOURCE_NAMES)
    selection.add_argument("--all", action="store_true", help="Try TextVQA, HH, WildJailbreak, VLGuard in order; continue on source errors")
    parser.add_argument("--data-root", type=Path, default=Path("data/step1"))
    parser.add_argument("--revision", help="HF branch/SHA; the first successful resolution is locked")
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args(argv)
    if args.max_workers < 1:
        parser.error("--max-workers must be positive")
    if args.all and args.revision is not None:
        parser.error("--revision applies to one HF source; use --dataset for a pinned override")
    try:
        if args.all:
            result = download_all(args.data_root, max_workers=args.max_workers)
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return 0 if result["status"] == "complete" else 1
        result = download_source(args.data_root, args.dataset, revision=args.revision, max_workers=args.max_workers)
    except KeyboardInterrupt:
        print("Download interrupted by user; completed files and resume state are preserved.", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"Download failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"dataset": result["source"], "status": "downloaded", "files": len(result["files"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
