"""Download official benchmark sources and reuse a frozen source lock on reruns."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.data_utils import (
    COMMIT_PATTERN, SOURCES, read_json, sha256_file, utc_now,
    validate_source_lock, write_json_atomic, write_text_atomic,
)


def resolve_github_commit(repo_id: str, revision: str) -> str:
    if COMMIT_PATTERN.fullmatch(revision):
        return revision
    # Use Git rather than the anonymous GitHub API, which is frequently rate limited.
    result = subprocess.run(
        ["git", "ls-remote", f"https://github.com/{repo_id}.git", f"refs/heads/{revision}"],
        check=True, capture_output=True, text=True, timeout=60,
    )
    lines = result.stdout.strip().splitlines()
    if len(lines) != 1:
        raise ValueError(f"Cannot resolve GitHub branch {repo_id}@{revision}; use a branch or full SHA")
    commit = lines[0].split()[0]
    if not COMMIT_PATTERN.fullmatch(commit):
        raise ValueError("GitHub returned an invalid commit SHA")
    return commit


def get_source_lock(path: Path, requests: dict, hf_api) -> dict:
    if path.exists():
        lock = validate_source_lock(read_json(path))
        for name in SOURCES:
            source = lock["sources"][name]
            requested = requests[name]
            if requested is not None and requested not in (source.get("requested_revision"), source["commit"]):
                raise ValueError("Revision differs from the existing lock; use a new data root")
        return lock

    resolved = {}
    for name, source in SOURCES.items():
        requested = requests[name] or "main"
        if source["kind"] == "github":
            commit = resolve_github_commit(source["repo_id"], requested)
        else:
            commit = hf_api.dataset_info(source["repo_id"], revision=requested).sha
        if not isinstance(commit, str) or not COMMIT_PATTERN.fullmatch(commit):
            raise ValueError(f"Invalid commit SHA returned for {name}")
        resolved[name] = {**source, "requested_revision": requested, "commit": commit}
        print(f"Pinned {name}: {commit}", flush=True)
    lock = {"schema_version": 1, "created_at": utc_now(), "sources": resolved}
    # Save before downloading images so interrupted runs retain the same revisions.
    write_json_atomic(path, lock)
    return lock


def fetch_text(url: str, destination: Path, *, expect_json: bool = False) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "mm-sft-lab/0.1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        text = response.read().decode("utf-8-sig")
    if expect_json:
        payload = json.loads(text)
        if not isinstance(payload, list) or not payload:
            raise ValueError(f"Expected a nonempty annotation list from {url}")
    write_text_atomic(destination, text)


def download(data_root: Path, requests: dict, max_workers: int) -> dict:
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as error:
        raise RuntimeError("Install data dependencies: python -m pip install -r requirements-data.txt") from error

    data_root = data_root.resolve()
    data_root.mkdir(parents=True, exist_ok=True)
    lock = get_source_lock(data_root / "sources.lock.json", requests, HfApi())
    sources = lock["sources"]
    annotations = data_root / "raw" / "siuo" / "annotations"
    commit = sources["siuo_annotations"]["commit"]
    base = f"https://raw.githubusercontent.com/sinwang20/SIUO/{commit}"
    for name in ("siuo_gen.json", "siuo_mcqa.json"):
        fetch_text(f"{base}/data/{name}", annotations / name, expect_json=True)
    fetch_text(f"{base}/README.md", annotations / "UPSTREAM_README.md")

    for name, relative in (
        ("siuo_images", "raw/siuo/images"),
        ("mssbench", "raw/mssbench"),
    ):
        source = sources[name]
        print(f"Downloading {source['repo_id']} at {source['commit']}", flush=True)
        snapshot_download(
            repo_id=source["repo_id"], repo_type="dataset", revision=source["commit"],
            local_dir=str(data_root / relative), max_workers=max_workers,
        )

    annotation_paths = [annotations / "siuo_gen.json", annotations / "siuo_mcqa.json"]
    mss_paths = [data_root / "raw" / "mssbench" / name for name in ("combined.json", "combine.json")]
    if not any(path.is_file() for path in mss_paths):
        raise ValueError("MSSBench snapshot contains neither combined.json nor combine.json")
    annotation_paths.extend(path for path in mss_paths if path.is_file())
    manifest = {
        "schema_version": 1, "completed_at": utc_now(), "sources": sources,
        "sources_lock_sha256": sha256_file(data_root / "sources.lock.json"),
        "annotation_sha256": {
            path.relative_to(data_root).as_posix(): sha256_file(path)
            for path in annotation_paths
        },
        "dependencies": {
            name: importlib.metadata.version(name)
            for name in ("huggingface-hub", "Pillow")
        },
    }
    write_json_atomic(data_root / "download_manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--siuo-revision", help="GitHub branch or full SHA; existing lock takes precedence")
    parser.add_argument("--siuo-images-revision", help="Hugging Face revision for SIUO images")
    parser.add_argument("--mssbench-revision", help="Hugging Face revision for MSSBench")
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args()
    if args.max_workers < 1:
        parser.error("--max-workers must be positive")
    try:
        download(args.data_root, {
            "siuo_annotations": args.siuo_revision,
            "siuo_images": args.siuo_images_revision,
            "mssbench": args.mssbench_revision,
        }, args.max_workers)
    except Exception as error:
        print(f"Download failed: {error}", file=sys.stderr)
        return 1
    print(f"Download complete: {args.data_root.resolve()}")
    print("Next: python scripts/prepare_datasets.py --data-root <same-directory>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
