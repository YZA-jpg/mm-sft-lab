"""Immutable identities, portable paths, and single-writer journals."""

import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


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
        if temporary:
            temporary.unlink(missing_ok=True)


def write_json(path, value):
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def write_jsonl(path, rows):
    atomic_text(path, "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows))


def read_journal(path, *, repair=False):
    path = Path(path)
    if not path.exists():
        return []
    lines, rows, offset = path.read_bytes().splitlines(keepends=True), [], 0
    for index, line in enumerate(lines):
        try:
            row = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            if repair and index == len(lines) - 1 and not line.endswith(b"\n"):
                with path.open("r+b") as stream:
                    stream.truncate(offset)
                break
            raise ValueError(f"Invalid journal line {index + 1}: {path}") from None
        if not isinstance(row, dict):
            raise ValueError("Journal row must be an object")
        rows.append(row)
        offset += len(line)
    else:
        if repair and lines and not lines[-1].endswith(b"\n"):
            with path.open("ab") as stream:
                stream.write(b"\n")
    return rows


def append(path, row):
    with Path(path).open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def latest(rows):
    result = {}
    for row in rows:
        key = row["job_id"]
        if key in result and row["attempt"] <= result[key]["attempt"]:
            raise ValueError(f"Duplicate/non-monotonic attempt: {key}")
        result[key] = row
    return result


@contextmanager
def lock(root, stage):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f".{stage}.lock"
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise ValueError(f"Stage locked: {path}; inspect owner, do not blindly remove locks") from None
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump({"pid": os.getpid(), "created_at": now()}, stream)
        yield
    finally:
        path.unlink(missing_ok=True)


def asset_path(root, name):
    relative = PurePosixPath(name)
    if not name or "\\" in name or ":" in name or relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe dataset asset path: {name}")
    root = Path(root).resolve()
    path = (root / Path(*relative.parts)).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Asset resolves outside dataset")
    return path


def code_identity():
    root = Path(__file__).resolve().parent
    files = {p.name: file_hash(p) for p in sorted(root.glob("*.py"))}
    return {"files": files, "sha256": digest(files)}
