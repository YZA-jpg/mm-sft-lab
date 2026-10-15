"""Read only validated prepared data and select whole benchmark units."""

from pathlib import Path, PurePosixPath

from .common import file_hash, read_json, read_jsonl
from .config import TASKS

INPUT_KEYS = {"id", "benchmark", "task", "source_scene_id", "query_index", "pair_id", "query", "media_paths", "options"}


def load_prepared(data_root, config):
    root = Path(data_root).resolve()
    prepared = root / "prepared"
    manifest = read_json(prepared / "manifest.json")
    if manifest.get("schema_version") != 1 or manifest.get("status") != "validated":
        raise ValueError("Data has not passed preparation")
    if config["require_locked_data"] and manifest.get("provenance_status") != "locked":
        raise ValueError("This configuration requires locked official data; mock/manual data must use a separate configuration")
    for name in ("inputs.jsonl", "evaluation.jsonl"):
        if manifest.get("output_sha256", {}).get(name) != file_hash(prepared / name):
            raise ValueError(f"Prepared file hash mismatch: {name}; rerun preparation")
    inputs = read_jsonl(prepared / "inputs.jsonl")
    labels = read_jsonl(prepared / "evaluation.jsonl")
    if not inputs or len({row["id"] for row in inputs}) != len(inputs) or len({row["id"] for row in labels}) != len(labels):
        raise ValueError("Empty data or duplicate prepared IDs")
    label_map = {row["id"]: row for row in labels}
    if set(label_map) != {row["id"] for row in inputs}:
        raise ValueError("Prepared input and evaluation IDs differ")
    image_map = {item["path"]: item["sha256"] for item in manifest["images"]}
    checked = set()
    pairs = {}
    for row in inputs:
        if set(row) - INPUT_KEYS:
            raise ValueError(f"Unexpected model-input fields in {row['id']}")
        if f"{row['benchmark']}/{row['task']}" not in TASKS or not isinstance(row["query"], str) or not row["query"].strip():
            raise ValueError(f"Invalid prepared input: {row['id']}")
        if not isinstance(row["media_paths"], list) or (row["benchmark"] != "wildjailbreak" and not row["media_paths"]):
            raise ValueError(f"No images in {row['id']}")
        if row["benchmark"] == "wildjailbreak" and row["media_paths"]:
            raise ValueError("WildJailbreak is text-only; do not attach placeholder images")
        for name in row["media_paths"]:
            relative = PurePosixPath(name)
            path = (root / name).resolve()
            if relative.is_absolute() or ".." in relative.parts or ":" in name or not path.is_relative_to(root):
                raise ValueError(f"Invalid image path: {name}")
            if name not in checked:
                if name not in image_map or not path.is_file() or file_hash(path) != image_map[name]:
                    raise ValueError(f"Image hash mismatch or missing file: {name}")
                checked.add(name)
        if row["benchmark"] == "mssbench":
            if not row.get("pair_id"):
                raise ValueError("MSSBench pair_id is missing")
            pairs.setdefault(row["pair_id"], []).append(row)
    for pair_id, members in pairs.items():
        if len(members) != 2 or {label_map[row["id"]].get("condition") for row in members} != {"safe", "unsafe"}:
            raise ValueError(f"Incomplete MSSBench pair: {pair_id}")
        if len({(row["task"], row["source_scene_id"]) for row in members}) != 1:
            raise ValueError(f"Inconsistent MSSBench pair identity: {pair_id}")
    selected = select_records(inputs, config["tasks"], config["per_task_limit"])
    missing_tasks = set(config["tasks"]) - {f"{row['benchmark']}/{row['task']}" for row in selected}
    if missing_tasks:
        raise ValueError(f"Prepared snapshot has no requests for: {sorted(missing_tasks)}")
    if not selected:
        raise ValueError("Selection is empty")
    return selected, [label_map[row["id"]] for row in selected], manifest


def select_records(rows, tasks, limit):
    selected = []
    for task in tasks:
        members = sorted((row for row in rows if f"{row['benchmark']}/{row['task']}" == task), key=lambda row: row["id"])
        units = {}
        for row in members:
            key = row["pair_id"] if row["benchmark"] == "mssbench" else row["id"]
            units.setdefault(key, []).append(row)
        keys = sorted(units)
        if limit is not None:
            keys = keys[:limit]
        for key in keys:
            selected.extend(units[key])
    return selected
