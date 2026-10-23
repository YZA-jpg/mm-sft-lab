"""Read-only per-dataset timing snapshots; no model, network, repair or source-data access."""

import argparse
import csv
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation_v2.common import atomic_text, digest, file_hash, latest, now, read_json, write_json
from evaluation_v2.metrics import timing_summary
from evaluation_v2.runner import generation_jobs
from evaluation_v2.status import snapshot


NOTES = [
    "Read-only snapshot of terminal job records; not a final score or an atomic live snapshot.",
    "Benchmark spans overlap during mixed-benchmark scheduling and must not be added.",
    "Span sums per-session first-start to last-finish; restart and stopped-job gaps between sessions are excluded.",
    "Active interval union is observed client job time, not exclusive GPU time.",
    "Job wall includes preparation/retries/backoff; sum of concurrent job latencies is not elapsed wall time.",
    "In-flight requests without terminal records are not included; use evaluate_v2.py status for stream progress.",
    "Expected judge jobs include generations not yet valid; pending planned jobs are not judge refusals.",
    "Incomplete pipeline sessions are listed; completed-pipeline wall alone omits failed/interrupted sessions.",
    "Serial benchmark boundary wall is exported separately; mixed schedules have no isolated benchmark wall.",
    "MSSBench chat/embodied are interleaved inside MSSBench; module spans are not isolated dataset wall.",
    "Model startup/shutdown is shared overhead, excluded from per-benchmark boundary wall and saved under services.",
]


def expected_jobs(records, config):
    result = {"generation": {}, "deepseek": {}, "wildguard": {}}
    for job in generation_jobs(records, config):
        meta = (job["benchmark"], job["split"])
        result["generation"][job["job_id"]] = meta
        if job["benchmark"] == "mmstar":
            continue
        phase = "wildguard" if job["benchmark"] == "wildjailbreak" else "deepseek"
        n = config["mmvet_judge_repeats"] if job["benchmark"] == "mmvet" else 1
        for index in range(n):
            result[phase][job["job_id"] + f"#j{index}"] = meta
    return result


def matches(meta, benchmark, split):
    return (benchmark == "ALL" or meta[0] == benchmark) and (split == "ALL" or meta[1] == split)


def build_timing(run_dir):
    root = Path(run_dir).resolve()
    manifest = read_json(root / "manifest.json")
    if digest(manifest["signature"]) != manifest["run_fingerprint"]:
        raise ValueError("Run fingerprint changed")
    for name, sha in manifest["snapshot_sha256"].items():
        if file_hash(root / name) != sha:
            raise ValueError("Frozen data snapshot changed: " + name)
    records = snapshot(root / "selection.jsonl")
    config = manifest["signature"]["config"]
    planned = expected_jobs(records, config)
    output, phase_sessions, benchmark_sessions = [], {}, {}
    schedule = config["execution"].get("benchmark_schedule", "mixed")
    for phase, expected in planned.items():
        history = snapshot(root / phase / "results.jsonl")
        events = snapshot(root / phase / "transport-attempts.jsonl")
        for row in [*history, *events]:
            if row.get("run_fingerprint") != manifest["run_fingerprint"] or row["job_id"] not in expected:
                raise ValueError("Mixed run or unexpected job in " + phase)
        for row in history:
            if (row["benchmark"], row["split"]) != expected[row["job_id"]]:
                raise ValueError("Journal benchmark mapping changed")
        current = latest(history)
        sessions = snapshot(root / phase / "sessions.jsonl")
        boundaries = snapshot(root / phase / "benchmark-sessions.jsonl")
        for event in boundaries:
            if event.get("run_fingerprint") != manifest["run_fingerprint"]:
                raise ValueError("Mixed run in benchmark timing boundaries")
        benchmark_sessions[phase] = boundaries
        completed_ids = {s["session_id"] for s in sessions if s.get("event") == "completed"}
        phase_sessions[phase] = {
            "completed_session_wall_seconds": sum(s["wall_seconds"] for s in sessions if s.get("event") == "completed"),
            "sessions_without_completed_event": sorted({s["session_id"] for s in sessions if s.get("event") == "started"} - completed_ids),
            "stage_lock_present": (root / phase / ".stage.lock").exists(),
        }
        if not expected:
            continue
        groups = [("ALL", "ALL")]
        for benchmark in sorted({b for b, split in expected.values()}):
            groups.append((benchmark, "ALL"))
            splits = sorted({split for b, split in expected.values() if b == benchmark})
            if len(splits) > 1:
                groups.extend((benchmark, split) for split in splits)
        for benchmark, split in groups:
            ids = {key for key, meta in expected.items() if matches(meta, benchmark, split)}
            rows = [r for r in history if r["job_id"] in ids]
            values = [r for key, r in current.items() if key in ids]
            valid = [r for r in values if r["status"] == "ok"]
            known = [r for r in valid if type((r.get("usage") or {}).get("completion_tokens")) is int]
            times = timing_summary(rows)
            terminals = [e for e in boundaries if e["benchmark"] == benchmark and e["event"] in ("completed", "failed")]
            starts = {(e["session_id"], e["benchmark"]) for e in boundaries if e["benchmark"] == benchmark and e["event"] == "started"}
            ends = {(e["session_id"], e["benchmark"]) for e in terminals}
            isolated = schedule == "serial" and benchmark != "ALL" and split == "ALL"
            first = min((r["job_started_epoch"] for r in rows), default=None)
            last = max((r["job_finished_epoch"] for r in rows), default=None)
            started_attempts = sum(e["job_id"] in ids for e in events)
            finished_attempts = sum(len(r.get("attempts", [])) for r in rows)
            output.append({
                "phase": phase, "benchmark": benchmark, "split": split,
                "expected_jobs": len(ids), "handled_unique_jobs": len(values),
                "valid_unique_jobs": len(valid), "error_unique_jobs": len(values) - len(valid),
                "unattempted_jobs": len(ids) - len(values), "processing_complete": len(values) == len(ids),
                "valid_coverage": len(valid) / len(ids) if ids else None,
                "truncated_valid_jobs": sum(r.get("finish_reason") == "length" for r in valid),
                "recorded_job_executions": len(rows),
                "isolated_benchmark_wall_seconds": sum(e["wall_seconds"] for e in terminals) if isolated and terminals else None,
                "isolated_timing_has_unfinished_session": bool(starts - ends) if isolated else None,
                "isolated_timing_has_failed_session": any(e["event"] == "failed" for e in terminals) if isolated else None,
                "observed_span_seconds_excluding_restart_gaps": times["span_seconds_excluding_restart_gaps"],
                "active_interval_union_seconds": times["active_interval_union_seconds"],
                "sum_job_wall_seconds": times["sum_job_wall_seconds"],
                "sum_http_attempt_seconds": times["sum_http_attempt_seconds"],
                "sum_backoff_seconds": times["sum_backoff_seconds"],
                "latency_p50_seconds": times["latency_p50_seconds"], "latency_p95_seconds": times["latency_p95_seconds"],
                "transport_attempts_started": started_attempts, "transport_attempts_with_terminal_metadata": finished_attempts,
                "failed_http_attempts_recorded": sum(a["status"] != "ok" for r in rows for a in r.get("attempts", [])),
                "latest_valid_usage_known_jobs": len(known),
                "latest_valid_completion_tokens": sum(r["usage"]["completion_tokens"] for r in known),
                "recorded_output_tokens_per_second": times["output_tokens_per_second"],
                "first_job_started_utc": datetime.fromtimestamp(first, timezone.utc).isoformat() if first is not None else None,
                "last_job_finished_utc": datetime.fromtimestamp(last, timezone.utc).isoformat() if last is not None else None,
            })
    pipeline = snapshot(root / "pipeline-sessions.jsonl")
    completed = {s["session_id"] for s in pipeline if s.get("event") == "completed"}
    services = [read_json(p) for p in sorted((root / "services").glob("*/launch.json"))]
    return {
        "schema_version": 1, "created_at": now(), "scope": "read_only_timing_snapshot_not_final_scores",
        "run_dir": str(root), "run_fingerprint": manifest["run_fingerprint"],
        "model_label": manifest["model_label"], "is_mock": manifest["is_mock"],
        "recorded_evaluation_code_sha256": manifest["signature"]["code_sha256"],
        "generation_parameters": config["generation"], "execution": config["execution"],
        "benchmark_schedule": schedule, "benchmark_sessions": benchmark_sessions,
        "rows": output, "phase_sessions": phase_sessions,
        "pipeline": {
            "completed_session_wall_seconds": sum(s["wall_seconds"] for s in pipeline if s.get("event") == "completed"),
            "sessions_without_completed_event": sorted({s["session_id"] for s in pipeline if s.get("event") == "started"} - completed),
            "pipeline_lock_present": (root / ".pipeline.lock").exists(),
        },
        "services": services, "notes": NOTES,
    }


def export_timing(run_dir, output_dir=None):
    report = build_timing(run_dir)
    output = Path(output_dir) if output_dir is not None else Path(run_dir) / (
        "timing-snapshot-" + report["created_at"].replace(":", "-"))
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "dataset-timing.json", report)
    text = io.StringIO(newline="")
    writer = csv.DictWriter(text, fieldnames=list(report["rows"][0]))
    writer.writeheader()
    writer.writerows(report["rows"])
    atomic_text(output / "dataset-timing.csv", text.getvalue())
    atomic_text(output / "README.md", "# Timing snapshot\n\n" + "\n".join("- " + n for n in NOTES) + "\n")
    return output, report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, help="New output directory; never overwrite an existing one")
    args = parser.parse_args(argv)
    output, report = export_timing(args.run_dir, args.output_dir)
    print(json.dumps({"scope": report["scope"], "output_dir": str(output.resolve()),
                      "csv": str((output / "dataset-timing.csv").resolve()),
                      "json": str((output / "dataset-timing.json").resolve()),
                      "phase_sessions": report["phase_sessions"], "pipeline": report["pipeline"]},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

