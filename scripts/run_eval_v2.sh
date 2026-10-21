#!/usr/bin/env bash
# Run from repository root; replace model path/label to evaluate a future SFT model.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-$PWD/.venv-data/bin/python}"
test -x "$PYTHON_BIN" || { echo "Set PYTHON_BIN to an environment containing Pillow"; exit 1; }
unset LD_LIBRARY_PATH LD_PRELOAD PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
exec "$PYTHON_BIN" -u scripts/evaluate_v2.py run "$@"
