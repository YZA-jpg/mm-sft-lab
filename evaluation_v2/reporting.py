"""Wide experimental tables, service/GPU timing, and protocol-aware comparisons."""

import csv
import io
from pathlib import Path
from statistics import mean

from .common import atomic_text, digest, read_json, read_journal, write_json
from .config import generation_parameters

MAIN_COLUMNS = [
    ("SIUO ASR@3", "siuo/ASR@3"), ("MSS Safe Acc", "mssbench/pooled/safe_acc"),
    ("MSS Unsafe Acc", "mssbench/pooled/unsafe_acc"), ("FigStep ASR@5", "figstep/ASR@5"),
    *(("HoliSafe " + kind, "holisafe/" + kind) for kind in ("SSS", "SUU", "USU", "UUU", "SSU")),
    ("WildJailbreak ASR", "wildjailbreak/ASR"), ("WildJailbreak ORR", "wildjailbreak/ORR"),
    ("MMStar Acc", "mmstar/Acc"), ("MathVista Acc", "mathvista/Acc"), ("MMVet Score", "mmvet/Score")]


def cell(value):
    if value is None:
        return "—"
    result = f"{value['percent']:.2f}%" if value["percent"] is not None else "N/A"
    return f"{result} ({value['x']} / {value['y']}; coverage {value['y']} / {value['expected']})"


def wide_table(reports, output):
    output = Path(output)
    columns = ["Model", *[title for title, _ in MAIN_COLUMNS]]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    for report in reports:
        label = report["model_label"] + (" [SIMULATION]" if report["is_mock"] else "")
        row = {"Model": label, **{title: cell(report["metrics"].get(key)) for title, key in MAIN_COLUMNS}}
        writer.writerow(row)
        lines.append("| " + " | ".join(str(row[name]).replace("|", "\\|").replace("\n", " ") for name in columns) + " |")
    lines += ["", "Percentages use valid y. Coverage is retained beside every result. HoliSafe SSS is RR; the other four types are ASR.",
              "MMVet x is a sum of scores in [0,1], not a correct-answer count. Protocol may still be provisional."]
    atomic_text(output / "main-table.md", "\n".join(lines) + "\n")
    atomic_text(output / "main-table.csv", stream.getvalue())


def resources(run_dir):
    root = Path(run_dir)
    sessions = read_journal(root / "pipeline-sessions.jsonl")
    completed = {r["session_id"] for r in sessions if r["event"] == "completed"}
    services = [read_json(path) for path in sorted((root / "services").glob("*/launch.json"))]
    samples = read_journal(root / "gpu-samples.jsonl")
    parsed = []
    for sample in samples:
        try:
            values = next(csv.reader([sample["csv"]], skipinitialspace=True))
            parsed.append({"utilization_percent": float(values[2]), "memory_used_MiB": float(values[3]),
                           "memory_free_MiB": float(values[4]), "power_W": float(values[5])})
        except (KeyError, ValueError, IndexError, StopIteration):
            continue
    return {"pipeline_wall_seconds_completed_sessions": sum(r["wall_seconds"] for r in sessions if r["event"] == "completed"),
            "interrupted_pipeline_sessions": [r["session_id"] for r in sessions if r["event"] == "started" and r["session_id"] not in completed],
            "services": services, "gpu": {"valid_samples": len(parsed), "observed_samples": len(samples),
                **{key: {"mean": mean(r[key] for r in parsed), "max": max(r[key] for r in parsed), "min": min(r[key] for r in parsed)}
                   for key in ("utilization_percent", "memory_used_MiB", "memory_free_MiB", "power_W") if parsed}},
            "scope": "nvidia-smi device-wide samples include other users/processes; not per-process memory or CUDA allocation. Pipeline wall includes service startup/cleanup; overlapping phases must not be added."}


def comparison_identity(report):
    # Labels, model paths and concurrency may differ, but metric protocol and endpoint defaults may not silently drift.
    config = report["config"]
    model = report["model_identity"]
    return {"schema_version": report["schema_version"], "protocol_id": report["protocol_id"],
            "protocol_status": report["protocol_status"], "is_mock": report["is_mock"],
            "data_manifest_sha256": report["data_manifest_sha256"], "selection_sha256": report["selection_sha256"],
            "code_sha256": report["code_sha256"], "judges": report["judges"],
            "repeats": config["repeats"], "mmvet_judge_repeats": config["mmvet_judge_repeats"],
            "target_generation_parameters": generation_parameters(config),
            "execution_policy": {k: v for k, v in config["execution"].items() if "concurrency" not in k},
            "target_generation_config_sha256": model.get("config_sha256", {}).get("generation_config.json"),
            "serving_runtime": model.get("serving_runtime"), "wildguard_identity": model.get("wildguard_identity")}


def compare(paths, output):
    reports = [read_json(Path(path) / "report/metrics.json" if Path(path).is_dir() else path) for path in paths]
    if not reports:
        raise ValueError("Supply at least one v2 report")
    first = comparison_identity(reports[0])
    for report in reports[1:]:
        value = comparison_identity(report)
        changes = [key for key in first if first[key] != value[key]]
        if changes:
            raise ValueError("Reports are not comparable under the frozen protocol: " + ", ".join(changes))
    output = Path(output)
    if (output / "comparison.json").exists():
        raise ValueError("Comparison already exists; use a new output directory")
    wide_table(reports, output)
    write_json(output / "comparison.json", {"comparison_signature": digest(first), "identity": first,
        "run_fingerprints": [r["run_fingerprint"] for r in reports], "reports": reports,
        "concurrency": [{k: v for k, v in r["config"]["execution"].items() if "concurrency" in k} for r in reports],
        "note": "Checkpoint-specific processor/chat templates may differ and are recorded in model_identity; compare carefully. This table does not establish a causal SFT effect."})
    return reports
