"""Small file contracts shared by the experiment stages."""

from __future__ import annotations

import hashlib
import json
import os
import socket
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_hash(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(text)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_json(path, value):
    atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def write_jsonl(path, rows):
    atomic_text(path, "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows))


def read_jsonl(path, *, repair_tail=False):
    path = Path(path)
    if not path.exists():
        return []
    payload = path.read_bytes()
    lines = payload.splitlines(keepends=True)
    rows, offset = [], 0
    for index, line in enumerate(lines):
        try:
            row = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            if repair_tail and index == len(lines) - 1 and not line.endswith(b"\n"):
                with path.open("r+b") as stream:
                    stream.truncate(offset)
                break
            raise ValueError(f"Invalid JSONL at {path}:{index + 1}") from None
        if not isinstance(row, dict):
            raise ValueError(f"JSONL rows must be objects: {path}:{index + 1}")
        rows.append(row)
        offset += len(line)
    else:
        if repair_tail and payload and not payload.endswith(b"\n"):
            with path.open("ab") as stream:
                stream.write(b"\n")
    return rows


def append_jsonl(path, row):
    with Path(path).open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


@contextmanager
def stage_lock(directory, stage):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f".{stage}.lock"
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise ValueError(f"Stage is locked: {path}; check the owner process before removing a stale lock") from None
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump({"pid": os.getpid(), "host": socket.gethostname(), "created_at": now()}, stream)
        yield
    finally:
        path.unlink(missing_ok=True)


def latest_attempts(rows):
    result = {}
    for row in rows:
        key = (row["id"], row["response_index"])
        if row.get("status") not in ("ok", "error"):
            raise ValueError(f"Invalid result status for {key}")
        if key in result and result[key]["status"] == "ok":
            raise ValueError(f"Result repeats a successful request: {key}")
        result[key] = row
    return result
