#!/usr/bin/env python3
"""CLI wrapper: validate ASE database quality."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mp_data_pipeline.validate import validate_db


if __name__ == "__main__":
    validate_db()
