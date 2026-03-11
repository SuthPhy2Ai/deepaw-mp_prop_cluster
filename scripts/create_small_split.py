#!/usr/bin/env python3
"""Create a small split for quick testing."""

import json
from pathlib import Path

# Load original split
split_file = Path("data/splits/split_iid_seed42.json")
with open(split_file) as f:
    split = json.load(f)

# Create small split (10% of data)
small_split = {
    "split_name": "iid_small",
    "seed": 42,
    "counts": {
        "train": 12390,
        "val": 1549,
        "test": 1549,
    },
    "mp_ids": {
        "train": split["mp_ids"]["train"][:12390],  # ~10% of 123,903
        "val": split["mp_ids"]["val"][:1549],       # ~10% of 15,487
        "test": split["mp_ids"]["test"][:1549],     # ~10% of 15,489
    }
}

# Save small split
small_split_file = Path("data/splits/split_iid_seed42_small.json")
with open(small_split_file, "w") as f:
    json.dump(small_split, f, indent=2)

print(f"Created small split: {small_split_file}")
print(f"  Train: {len(small_split['mp_ids']['train'])} samples")
print(f"  Val: {len(small_split['mp_ids']['val'])} samples")
print(f"  Test: {len(small_split['mp_ids']['test'])} samples")
print(f"\nEstimated data loading time: {len(small_split['mp_ids']['train']) * 0.651 / 60:.1f} minutes")
