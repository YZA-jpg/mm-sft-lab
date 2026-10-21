"""Import eight native benchmarks into a portable, hashed v2 snapshot."""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from evaluation_v2.common import file_hash, write_json
from evaluation_v2.config import BENCHMARKS
from evaluation_v2.data import (Writer, import_legacy, import_figstep, import_holisafe, import_mmstar,
                                import_mathvista, import_mmvet, literal_strings)
from evaluation_v2.sources import verify_sources

EXPECTED = {"siuo": 167, "mssbench": 1960, "figstep": 500, "holisafe": 4031,
            "wildjailbreak": 2210, "mmstar": 1500, "mathvista": 1000, "mmvet": 218}


def prepare(raw, legacy, output, names):
    raw, legacy = Path(raw).resolve(), Path(legacy).resolve()
    provenance = verify_sources(raw)
    required_sources = {name for name in names if name not in ("siuo", "mssbench", "wildjailbreak")}
    required_sources |= {"upstream/" + name for name in names if name in ("holisafe", "mathvista", "mmvet")}
    if "figstep" in names:
        required_sources.add("upstream/adashield")
    if "wildjailbreak" in names:
        required_sources.add("upstream/wildguard")
    if required_sources - set(provenance):
        raise ValueError("Missing verified source inventories: " + str(sorted(required_sources - set(provenance))))
    writer = Writer(output)
    rows, errors, is_mock = [], {}, False
    legacy_names = set(names) & {"siuo", "mssbench", "wildjailbreak"}
    if legacy_names:
        old, is_mock = import_legacy(writer, legacy, legacy_names)
        templates = literal_strings(REPO / "src/mm_sft_lab/benchmarks.py")
        writer.sources[str(REPO / "src/mm_sft_lab/benchmarks.py")] = file_hash(REPO / "src/mm_sft_lab/benchmarks.py")
        for row in old:
            if row["benchmark"] == "mssbench":
                row["query"] = templates["CHAT_IF" if row["split"] == "chat" else "EMBODIED_IF"] + row["query"]
        rows.extend(old)
        rubrics = literal_strings(REPO / "src/mm_sft_lab/judge.py")
        for name in legacy_names & {"siuo", "mssbench"}:
            writer.prompt(name, rubrics["SIUO_RUBRIC" if name == "siuo" else "MSS_RUBRIC"], REPO / "src/mm_sft_lab/judge.py")
    adapters = {"figstep": lambda: import_figstep(writer, raw / "figstep", raw / "upstream/adashield/system_prompts.py"),
                "holisafe": lambda: import_holisafe(writer, raw / "holisafe"),
                "mmstar": lambda: import_mmstar(writer, raw / "mmstar"),
                "mathvista": lambda: import_mathvista(writer, raw / "mathvista"),
                "mmvet": lambda: import_mmvet(writer, raw / "mmvet")}
    prompt_specs = {"holisafe": ("upstream/holisafe/eval/eval_by_openai_parallel_v2.py", "PROMPT_TEMPLATE", False),
                    "mathvista": ("upstream/mathvista/evaluation/prompts/ext_ans.py", "demo_prompt", False),
                    "mmvet": ("upstream/mmvet/mm-vet_evaluator.ipynb", "prompt", True),
                    "wildjailbreak": ("upstream/wildguard/wildguard/utils.py", "WILDGUARD_INPUT_FORMAT", False)}
    for name in names:
        try:
            if name in adapters:
                values = adapters[name]()
                if not is_mock and len(values) != EXPECTED[name]:
                    raise ValueError(f"Expected {EXPECTED[name]} {name} records, got {len(values)}; verify version/split")
                rows.extend(values)
            if name in prompt_specs:
                relative, variable, notebook = prompt_specs[name]
                source = raw / relative
                writer.prompt("wildguard" if name == "wildjailbreak" else name,
                              literal_strings(source, notebook=notebook)[variable], source)
        except Exception as error:
            errors[name] = f"{type(error).__name__}: {error}"
    counts = Counter(r["benchmark"] for r in rows)
    for name in names:
        if not is_mock and counts[name] != EXPECTED[name]:
            errors.setdefault(name, f"Incomplete benchmark: {counts[name]} / {EXPECTED[name]}")
    if errors:
        write_json(Path(output) / "preparation-errors.json", errors)
        raise ValueError("Preparation incomplete; sources retained: " + json.dumps(errors, ensure_ascii=False))
    return writer.finish(rows, is_mock=is_mock, provenance=provenance)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=Path("data/eval-v2/raw"))
    parser.add_argument("--legacy-root", type=Path, default=Path("data/step1"))
    parser.add_argument("--output", type=Path, default=Path("data/eval-v2/prepared"))
    parser.add_argument("--benchmarks", nargs="+", choices=BENCHMARKS, default=list(BENCHMARKS))
    args = parser.parse_args()
    try:
        result = prepare(args.raw_root, args.legacy_root, args.output, args.benchmarks)
    except Exception as error:
        print(f"Preparation failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"status": "prepared", "is_mock": result["is_mock"], "counts": result["counts"],
                      "split_counts": result["split_counts"], "unique_images": len(result["images"]),
                      "manifest": str(args.output / "manifest.json")}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
