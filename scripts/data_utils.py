"""Small shared helpers for reproducible benchmark preparation."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any


SOURCES = {
    "siuo_annotations": {"kind": "github", "repo_id": "sinwang20/SIUO"},
    "siuo_images": {"kind": "huggingface", "repo_id": "sinwang/SIUO"},
    "mssbench": {"kind": "huggingface", "repo_id": "kzhou35/mssbench"},
}
COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")


def validate_source_lock(lock: Any) -> dict:
    if not isinstance(lock, dict) or lock.get("schema_version") != 1:
        raise ValueError("Invalid source lock schema")
    sources = lock.get("sources")
    if not isinstance(sources, dict) or set(sources) != set(SOURCES):
        raise ValueError("Unexpected source names in lock")
    for name, expected in SOURCES.items():
        source = sources[name]
        if not isinstance(source, dict) or any(source.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Unexpected source in lock: {name}")
        commit = source.get("commit")
        if not isinstance(commit, str) or not COMMIT_PATTERN.fullmatch(commit):
            raise ValueError(f"Invalid pinned revision: {name}")
    return lock


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8-sig") as handle:
        return json.load(handle)


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(text)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_json_atomic(path: Path, value: Any) -> None:
    write_text_atomic(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def relative_media_path(value: Any) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Expected a nonempty relative image path, got {value!r}")
    normalized = value.replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or ":" in normalized or ".." in path.parts:
        raise ValueError(f"Image path must stay inside its dataset: {value!r}")
    return Path(*path.parts)
