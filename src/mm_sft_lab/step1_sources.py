"""Download the active Step 1 sources; never execute dataset scripts."""

import json
import os
import re
import shutil
import stat
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

from .common import digest, file_hash, now, read_json, stage_lock, write_json

HF_SOURCES = {
    "hh": {"repo_id": "Anthropic/hh-rlhf", "patterns": ["harmless-base/*.jsonl.gz", "README.md"], "license": "MIT"},
    "wildjailbreak": {"repo_id": "allenai/wildjailbreak", "patterns": ["eval/*.tsv", "README.md"], "license": "ODC-BY"},
    "vlguard": {"repo_id": "ys-zong/VLGuard", "patterns": ["train.json", "train.zip", "README.md"], "license": "MIT"},
}
HF_REQUIRED_FILES = {
    "hh": {"harmless-base/train.jsonl.gz", "harmless-base/test.jsonl.gz"},
    "wildjailbreak": {"eval/eval.tsv"},
    "vlguard": {"train.json", "train.zip"},
}
TEXTVQA_URLS = {
    **{f"TextVQA_0.5.1_{split}.json": f"https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_{split}.json"
       for split in ("train", "val", "test")},
    "train_val_images.zip": "https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip",
    "test_images.zip": "https://dl.fbaipublicfiles.com/textvqa/images/test_images.zip",
}
# Downloading VLGuard does not yet enable its training-data conversion.
PREPARATION_SOURCE_NAMES = ("textvqa", "hh", "wildjailbreak")
SOURCE_NAMES = (*PREPARATION_SOURCE_NAMES, "vlguard")


def inside(root, name):
    root = Path(root).resolve()
    if not isinstance(name, str) or not name or "\\" in name:
        raise ValueError(f"Invalid relative path: {name!r}")
    relative = PurePosixPath(name)
    candidate = (root / name).resolve()
    if relative.is_absolute() or ".." in relative.parts or ":" in name or not candidate.is_relative_to(root):
        raise ValueError(f"Path escapes data root: {name}")
    return candidate


def fetch_file(url, path):
    """Resume a CDN download only when the server confirms the requested offset."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "mm-sft-lab/0.1", "Accept-Encoding": "identity"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120) as response:
        append = response.status == 206
        if append and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
            raise ValueError("CDN returned an inconsistent resume offset")
        if response.status not in (200, 206):
            raise ValueError(f"Unexpected CDN status: {response.status}")
        expected = response.headers.get("Content-Length")
        received = 0
        next_progress = 256 * 1024 * 1024
        with partial.open("ab" if append else "wb") as stream:
            while chunk := response.read(1024 * 1024):
                stream.write(chunk)
                received += len(chunk)
                if received >= next_progress:
                    print(f"{path.name}: received {received // (1024 * 1024)} MiB this session", flush=True)
                    next_progress += 256 * 1024 * 1024
        if expected is not None and received != int(expected):
            raise ValueError(f"Incomplete download: {path.name}; rerun to resume")
    if not partial.stat().st_size:
        raise ValueError("Downloaded an empty file")
    os.replace(partial, path)


def verify_source(root, name, *, require_locked=True):
    if name not in SOURCE_NAMES:
        raise ValueError(f"Unknown dataset: {name}")
    base = Path(root).resolve() / "raw" / name
    manifest = read_json(base / "download_manifest.json")
    if manifest.get("schema_version") != 1 or manifest.get("source") != name or not manifest.get("files"):
        raise ValueError(f"Invalid download manifest: {name}")
    lock = read_json(base / "source.lock.json")
    if digest(lock) != manifest.get("source_lock_sha256"):
        raise ValueError(f"Source lock changed: {name}")
    if require_locked and manifest.get("provenance") != "locked":
        raise ValueError(f"Source is not locked: {name}")
    required = set(TEXTVQA_URLS) if name == "textvqa" else HF_REQUIRED_FILES[name]
    if not required.issubset(manifest["files"]):
        raise ValueError(f"Download manifest is missing required assets: {name}")
    if manifest.get("provenance") == "locked":
        if name == "textvqa":
            if (lock.get("urls") != TEXTVQA_URLS or lock.get("version") != "0.5.1" or
                    lock.get("completed_files") != manifest["files"]):
                raise ValueError("Unexpected TextVQA source")
        elif name in HF_SOURCES:
            if lock.get("repo_id") != HF_SOURCES[name]["repo_id"] or not re.fullmatch(r"[0-9a-f]{40}", lock.get("commit", "")):
                raise ValueError(f"Unexpected HF source: {name}")
    for relative, expected in manifest["files"].items():
        path = inside(base, relative)
        if not path.is_file() or file_hash(path) != expected["sha256"]:
            raise ValueError(f"Raw source hash mismatch: {name}/{relative}")
    return manifest


def download_source(root, name, *, revision=None, max_workers=4, hf_api=None, snapshot=None, fetch=fetch_file):
    if name not in SOURCE_NAMES:
        raise ValueError(f"Unknown dataset: {name}; select one of {', '.join(SOURCE_NAMES)}")
    base = Path(root).resolve() / "raw" / name
    with stage_lock(base, "download"):
        manifest_path, lock_path = base / "download_manifest.json", base / "source.lock.json"
        if manifest_path.exists():
            result = verify_source(root, name)
            lock = read_json(lock_path)
            if revision and revision not in (lock.get("commit"), lock.get("requested_revision")):
                raise ValueError("Revision changed; use a new data root")
            return result
        lock = read_json(lock_path) if lock_path.exists() else None
        if name == "textvqa":
            if revision is not None:
                raise ValueError("TextVQA uses the fixed official v0.5.1 URLs")
            expected = {"source": name, "version": "0.5.1", "urls": TEXTVQA_URLS, "license": "CC-BY-4.0"}
            if lock and any(lock.get(k) != v for k, v in expected.items()):
                raise ValueError("TextVQA lock does not match the configured source")
            lock = lock or {**expected, "created_at": now(), "completed_files": {}}
            write_json(lock_path, lock)
            for relative, url in TEXTVQA_URLS.items():
                path = inside(base, relative)
                saved = lock["completed_files"].get(relative)
                if saved:
                    if not path.is_file() or file_hash(path) != saved["sha256"]:
                        raise ValueError(f"Completed asset changed: {relative}; use a new data root")
                    continue
                print(f"Downloading official TextVQA {relative}", flush=True)
                fetch(url, path)
                lock["completed_files"][relative] = {"sha256": file_hash(path), "bytes": path.stat().st_size}
                write_json(lock_path, lock)
            files = lock["completed_files"]
        else:
            if hf_api is None or snapshot is None:
                from huggingface_hub import HfApi, snapshot_download
                hf_api, snapshot = HfApi(), snapshot_download
            spec = HF_SOURCES[name]
            if lock:
                if lock.get("repo_id") != spec["repo_id"] or revision and revision not in (lock["commit"], lock["requested_revision"]):
                    raise ValueError("HF source/revision changed; use a new data root")
            else:
                requested = revision or "main"
                commit = hf_api.dataset_info(spec["repo_id"], revision=requested).sha
                if not re.fullmatch(r"[0-9a-f]{40}", commit or ""):
                    raise ValueError("HF returned an invalid commit")
                lock = {"source": name, "repo_id": spec["repo_id"], "license": spec["license"],
                        "commit": commit, "requested_revision": requested, "created_at": now()}
                write_json(lock_path, lock)
            print(f"Pinned {spec['repo_id']} at {lock['commit']}", flush=True)
            snapshot(repo_id=spec["repo_id"], repo_type="dataset", revision=lock["commit"],
                     allow_patterns=spec["patterns"], local_dir=str(base), max_workers=max_workers)
            if name == "hh":
                paths = sorted(base.glob("harmless-base/*.jsonl.gz"))
            elif name == "wildjailbreak":
                paths = sorted(base.glob("eval/*.tsv"))
            else:
                paths = [base / relative for relative in sorted(HF_REQUIRED_FILES[name])]
            if name == "hh" and {p.name for p in paths} != {"train.jsonl.gz", "test.jsonl.gz"}:
                raise ValueError("HH snapshot is missing harmless-base train/test")
            if name == "wildjailbreak" and [p.name for p in paths] != ["eval.tsv"]:
                raise ValueError("Expected the official eval/eval.tsv; do not download training pairs")
            if any(not path.is_file() or path.stat().st_size == 0 for path in paths):
                raise ValueError(f"HF snapshot is missing required assets or contains empty files: {name}")
            if (base / "README.md").is_file():
                paths.append(base / "README.md")
            files = {p.relative_to(base).as_posix(): {"sha256": file_hash(p), "bytes": p.stat().st_size} for p in paths}
        result = {"schema_version": 1, "source": name, "provenance": "locked", "completed_at": now(),
                  "source_lock_sha256": digest(lock), "files": files}
        write_json(manifest_path, result)
        return result


def extract_images(archive, destination):
    """Extract ZIP assets within one directory, rejecting symlinks and traversal."""
    archive, destination = Path(archive), Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    signature = file_hash(archive)
    marker = destination / ".extracted.json"
    with zipfile.ZipFile(archive) as bundle:
        members = bundle.infolist()
        targets, seen = [], set()
        for item in members:
            relative = item.filename.rstrip("/")
            if not relative:
                continue
            path = inside(destination, relative)
            if path in seen:
                raise ValueError("ZIP contains duplicate destination paths")
            seen.add(path)
            if stat.S_ISLNK(item.external_attr >> 16):
                raise ValueError("ZIP symlinks are not supported")
            targets.append((item, path))
        if marker.exists() and read_json(marker).get("archive_sha256") == signature and all(
            path.is_dir() if item.is_dir() else path.is_file() and path.stat().st_size == item.file_size
            for item, path in targets
        ):
            return
        for item, path in targets:
            if item.is_dir():
                path.mkdir(parents=True, exist_ok=True)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(item) as source, path.open("wb") as target:
                    shutil.copyfileobj(source, target, length=1024 * 1024)
        write_json(marker, {"archive_sha256": signature, "files": len(targets)})
