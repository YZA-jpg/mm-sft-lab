"""One-card vLLM concurrency measurement; no DeepSeek or WildGuard judging calls."""
import argparse
import os
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation_v2.config import load
from evaluation_v2.profiling import sweep


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/eval_v2.full.json"))
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--vllm-python", default=sys.executable)
    parser.add_argument("--gpu", default="7")
    parser.add_argument("--gpu-memory-utilization", type=float)
    parser.add_argument("--target-kv-cache-gib", type=float)
    parser.add_argument("--levels", nargs="+", type=int, default=[1, 4, 8, 16])
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--limit", type=int, default=4, help="Units per benchmark split; MSS unit is a complete pair")
    args = parser.parse_args()
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    if os.name == "posix":
        signal.signal(signal.SIGTERM, interrupted)
    sweep(args.data_root, args.output, load(args.config), args.model_path, levels=args.levels, rounds=args.rounds,
          limit=args.limit, gpu=args.gpu, vllm_python=args.vllm_python, adapter=args.adapter_path,
          memory=args.gpu_memory_utilization, target_kv_cache_gib=args.target_kv_cache_gib)
    print(args.output / "sweep.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
