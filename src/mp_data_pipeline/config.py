"""MP Data Pipeline Configuration."""
import os
from pathlib import Path

# API
MP_API_KEY = os.environ.get("MP_API_KEY", "")

# Paths
PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
DB_DIR = DATA_DIR / "db"
LOG_DIR = PROJECT_DIR / "logs"
CHECKPOINT_DIR = DATA_DIR / "checkpoints"
RAW_JSONL_PATH = RAW_DIR / "summary_all_merged.jsonl.gz"
DB_PATH = DB_DIR / "mp_materials.db"
TEST_DB_PATH = DB_DIR / "mp_test_1000.db"

# Download settings
CHUNK_SIZE = 500
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds
