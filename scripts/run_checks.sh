#!/usr/bin/env bash
# Run one explicitly selected phase. No environment installation or API judging.
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$repo_root"
phase="${1:-}"
if [[ -z "$phase" ]]; then
  echo "Usage: bash scripts/run_checks.sh {plan|smoke|batch|small} [--resume] [--retry-errors]" >&2
  exit 1
fi
shift
case "$phase" in plan|smoke|batch|small) ;; *) echo "Unknown phase: $phase" >&2; exit 1 ;; esac
resume_args=()
retry_errors=false
resume=false
for flag in "$@"; do
  case "$flag" in
    --resume) resume=true ;;
    --retry-errors) retry_errors=true ;;
    *) echo "Unknown option: $flag" >&2; exit 1 ;;
  esac
done
if $retry_errors && ! $resume; then
  echo "--retry-errors requires --resume" >&2
  exit 1
fi
if $resume; then resume_args+=(--resume); fi
if $retry_errors; then resume_args+=(--retry-errors); fi

python_bin="${PYTHON_BIN:-$repo_root/.venv/bin/python}"
data_root="${DATA_ROOT:-$repo_root/data/imported}"
run_root="${RUN_ROOT:-$repo_root/runs/qwen3-vl-4b-checks}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
if ! "$python_bin" --version; then
  echo "Set PYTHON_BIN to a working project inference environment. See docs/smoke-tests.md." >&2
  exit 1
fi
mkdir -p -- "$run_root/logs"
log_file="$run_root/logs/$phase-$(date -u +%Y%m%dT%H%M%SZ)-$$.log"
exec > >(tee -a "$log_file") 2>&1
echo "Phase: $phase | GPU visibility: $CUDA_VISIBLE_DEVICES | Python: $python_bin"
echo "Data: $data_root | Runs: $run_root | Log: $log_file"

lab() { "$python_bin" scripts/lab.py "$@"; }
run_config() {
  local config="$1" directory="$2"
  lab plan --config "$config" --data-root "$data_root"
  lab generate --config "$config" --data-root "$data_root" --run-dir "$directory" "${resume_args[@]}"
  lab validate-run --run-dir "$directory"
}
compare_smoke() {
  lab compare-batches --before "$run_root/smoke-b1" --after "$run_root/smoke-b4" --output "$run_root/batch-comparison"
}

case "$phase" in
  plan)
    for config in configs/baseline_smoke.json configs/baseline_smoke_batch4.json configs/baseline_small.json; do
      echo "Selection: $config"
      lab plan --config "$config" --data-root "$data_root"
    done
    ;;
  smoke)
    lab preflight --config configs/baseline_smoke.json
    run_config configs/baseline_smoke.json "$run_root/smoke-b1"
    ;;
  batch)
    lab validate-run --run-dir "$run_root/smoke-b1"
    lab preflight --config configs/baseline_smoke_batch4.json
    run_config configs/baseline_smoke_batch4.json "$run_root/smoke-b4"
    compare_smoke
    ;;
  small)
    # Recompute from current journals; a stale comparison file cannot authorize this phase.
    compare_smoke
    lab preflight --config configs/baseline_small.json
    run_config configs/baseline_small.json "$run_root/small-b4"
    ;;
esac
