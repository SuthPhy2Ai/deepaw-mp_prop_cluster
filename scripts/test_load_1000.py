#!/usr/bin/env python3
"""CLI wrapper: build a 1000-record test ASE DB."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mp_data_pipeline.test_load_1000 import main


if __name__ == "__main__":
    main()
