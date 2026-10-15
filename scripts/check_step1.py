"""Run offline unit tests and a synthetic end-to-end Step 1 pipeline."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
from scripts.make_step1_demo import make_step1_demo
from scripts.make_vlguard_demo import make_vlguard_demo
from mm_sft_lab.data_audit import audit_data
from mm_sft_lab.common import write_json
from mm_sft_lab.config import load_config, load_judge_config
from mm_sft_lab.diagnostics import validate_run
from mm_sft_lab.judge import judge_run
from mm_sft_lab.manual_judge import export_review
from mm_sft_lab.metrics import score_run
from mm_sft_lab.runner import generate
from mm_sft_lab.training_data import load_train_config, train_plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New empty directory; no existing artifacts are deleted")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("Output directory is not empty; select a new directory")
    try:
        subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"], cwd=REPO, check=True)
        data = output / "data"
        prepared = make_step1_demo(data)
        vlguard = make_vlguard_demo(data)
        data_audit = audit_data(data, REPO / "configs/train", vlguard_root=data / "vlguard")
        write_json(output / "mock_data_audit.json", data_audit)
        if data_audit["violations"]:
            raise ValueError("Synthetic three-group data audit failed")
        config = {**load_config(REPO / "configs/step1_smoke.json"), "backend": "mock", "require_locked_data": False}
        write_json(output / "mock_eval_config.json", config)
        run, judge = output / "run", output / "judge"
        generate(config, data, run)
        checks = validate_run(run)
        unjudged, _ = score_run(run)
        export_review(run, output / "review_template.jsonl")
        judgments = judge_run(run, judge, load_judge_config(REPO / "configs/judge.example.json"), mock=True)
        scored, _ = score_run(run, judge)
        plans = {}
        for name in ("textvqa", "hh", "vlguard"):
            train_config = {**load_train_config(REPO / f"configs/train/{name}_lora.json"), "require_locked_data": False}
            plans[name] = train_plan(data / "vlguard" if name == "vlguard" else data, train_config)
            write_json(output / f"mock_train_{name}.json", train_config)
        if not checks["complete"] or not scored["complete"] or not scored["is_mock"]:
            raise ValueError("Synthetic pipeline completeness/mock guards failed")
        result = {"status": "passed", "scope": "unit tests and synthetic pipeline only", "is_mock": True,
                  "real_training_verified": False, "real_generation_verified": False,
                  "training_counts": {**prepared["audit"]["training_counts"], **vlguard["audit"]["training_counts"]},
                  "evaluation_counts": prepared["request_counts"],
                  "judge_requests": judgments["successful_requests"], "pending_before_judging": not unjudged["complete"],
                  "output": str(output)}
        write_json(output / "summary.json", result)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    except Exception as error:
        print(f"Offline check failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
