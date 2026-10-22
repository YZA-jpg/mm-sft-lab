"""Read-only live progress from complete journal lines, without API calls or lock removal."""

import json
from collections import Counter
from pathlib import Path

from .common import latest, now, read_json
from .config import generation_parameters


def snapshot(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open("rb") as stream:
        payload = stream.read(path.stat().st_size)
    # A writer may currently be appending the final line. Never repair/truncate its file.
    return [json.loads(line) for line in payload[:payload.rfind(b"\n") + 1].splitlines()]


def status(run_dir):
    root = Path(run_dir).resolve()
    manifest = read_json(root / "manifest.json")
    config = manifest["signature"]["config"]
    selected = snapshot(root / "selection.jsonl")
    expected_generation = sum(config["repeats"][r["benchmark"]] for r in selected)
    expected_cloud = sum(config["repeats"][r["benchmark"]] * (config["mmvet_judge_repeats"] if r["benchmark"] == "mmvet" else 1)
                         for r in selected if r["benchmark"] not in ("wildjailbreak", "mmstar"))
    expected_guard = sum(r["benchmark"] == "wildjailbreak" for r in selected)
    phases = {}
    for phase, expected in (("generation", expected_generation), ("deepseek", expected_cloud), ("wildguard", expected_guard)):
        folder = root / phase
        rows = latest(snapshot(folder / "results.jsonl"))
        recent = max(rows.values(), key=lambda r: r["job_finished_epoch"], default={})
        lock = folder / ".stage.lock"
        try:
            owner = read_json(lock) if lock.exists() else None
        except (ValueError, FileNotFoundError):
            owner = {"note": "Lock was being created/removed during this snapshot"}
        phases[phase] = {"expected_full_dataset_jobs": expected, "handled": len(rows),
            "valid": sum(r["status"] == "ok" for r in rows.values()), "remaining_without_result": expected - len(rows),
            "status_counts": dict(Counter(r.get("error_kind") or "valid" for r in rows.values())),
            "by_benchmark": dict(Counter(r["benchmark"] for r in rows.values())), "lock_owner": owner,
            "latest_job": recent.get("job_id"), "latest_finished_epoch": recent.get("job_finished_epoch")}
        if phase == "generation":
            phases[phase].update(finish_reasons=dict(Counter(r.get("finish_reason") for r in rows.values())),
                truncated=sum(r["status"] == "ok" and r.get("finish_reason") == "length" for r in rows.values()))
    streams = []
    for path in (root / "generation/streams").glob("*/attempt-*/progress.json"):
        if (path.parent / "summary.json").exists():
            continue
        try:
            identity = read_json(path.parent.parent / "identity.json")
            progress = read_json(path)
        except (ValueError, FileNotFoundError):
            continue
        if identity.get("run_fingerprint") == manifest["run_fingerprint"]:
            streams.append({"job_id": identity["job_id"], "attempt": path.parent.name, **progress})
    return {"at": now(), "run_dir": str(root), "run_fingerprint": manifest["run_fingerprint"],
            "is_mock": manifest["is_mock"], "target_generation_parameters": generation_parameters(config), "phases": phases,
            "incomplete_streams": sorted(streams, key=lambda row: row.get("updated_at", ""))[-16:],
            "stream_progress_scope": "Last saved progress of streams without a terminal capture summary; presence alone does not prove their processes are alive.",
            "scope": "Read-only live snapshot, not a signed final score. Remaining judge jobs include generation failures that cannot yet be judged. Lock presence is not proof that its owner is alive."}
