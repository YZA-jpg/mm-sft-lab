"""Run the project CLI from a checkout without an editable install."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mm_sft_lab.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
