"""Utility functions for MP data pipeline."""
import json
import logging
import time
from pathlib import Path
from typing import Any, List, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


def setup_logging(level=logging.INFO):
    """Configure logging."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    # Force flush after each log
    import sys
    for handler in logging.root.handlers:
        handler.flush = lambda: sys.stdout.flush() if hasattr(sys.stdout, 'flush') else None


def save_checkpoint(data: list[dict], path: Path, tag: str):
    """Save intermediate results as JSON checkpoint."""
    path.mkdir(parents=True, exist_ok=True)
    fpath = path / f"checkpoint_{tag}.json"
    # Convert non-serializable types
    serializable = []
    for d in data:
        s = {}
        for k, v in d.items():
            if isinstance(v, np.ndarray):
                s[k] = v.tolist()
            elif hasattr(v, "as_dict"):  # pymatgen objects
                s[k] = v.as_dict()
            else:
                s[k] = v
        serializable.append(s)
    with open(fpath, "w") as f:
        json.dump(serializable, f)
    logger.info(f"Checkpoint saved: {fpath} ({len(data)} records)")


def load_checkpoint(path: Path, tag: str) -> Optional[List[dict]]:
    """Load checkpoint if exists."""
    fpath = path / f"checkpoint_{tag}.json"
    if fpath.exists():
        with open(fpath) as f:
            data = json.load(f)
        logger.info(f"Checkpoint loaded: {fpath} ({len(data)} records)")
        return data
    return None


def retry_with_backoff(func, max_retries=3, base_delay=5):
    """Retry a function with exponential backoff."""
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt)
            logger.warning(f"Attempt {attempt+1} failed: {e}. Retrying in {delay}s...")
            time.sleep(delay)
