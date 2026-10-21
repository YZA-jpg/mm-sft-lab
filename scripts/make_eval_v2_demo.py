"""Execute an eight-benchmark simulation without sockets, models, keys or paid calls."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation_v2.config import validate
from evaluation_v2.demo import FakeHTTP, make_demo
from evaluation_v2.metrics import export_report
from evaluation_v2.runner import initialize, stage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data, run = args.output / "data", args.output / "run"
    make_demo(data)
    config = validate({"execution": {"generation_concurrency": 4, "deepseek_concurrency": 3}})
    initialize(data, run, config, {"kind": "mock", "label": "DEMO ONLY"})
    fake = FakeHTTP()
    for name in ("generation", "deepseek", "wildguard"):
        stage(run, name, client_factory=fake.factory)
    count = len(fake.calls)
    for name in ("generation", "deepseek", "wildguard"):
        stage(run, name, client_factory=fake.factory)
    if len(fake.calls) != count:
        raise AssertionError("Resume repeated a completed request")
    report = export_report(run)
    print(json.dumps({"is_mock": report["is_mock"], "complete": report["complete"], "fake_calls": count,
                      "resume_new_calls": len(fake.calls) - count, "table": str(run / 'report/table.md')}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
