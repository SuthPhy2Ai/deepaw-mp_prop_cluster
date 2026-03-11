#!/usr/bin/env python3
"""Test data loading speed."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mp_data_pipeline.config import DB_PATH
from mp_data_pipeline.ml.dataset import AseGraphMultitaskDataset
from mp_data_pipeline.ml.splits import load_split

print("Loading split...")
split = load_split(Path("data/splits/split_iid_seed42.json"))
print(f"Train samples: {len(split['train'])}")

print("\nBuilding dataset (first 100 samples)...")
start = time.time()

train_ds = AseGraphMultitaskDataset(
    db_path=DB_PATH,
    mp_ids=split["train"][:100],  # Only first 100 samples
    cutoff=6.0,
    max_neighbors=24,
)

elapsed = time.time() - start
print(f"Time: {elapsed:.2f} seconds")
print(f"Samples: {len(train_ds)}")
print(f"Time per sample: {elapsed/len(train_ds):.3f} seconds")

print("\nFirst sample:")
sample = train_ds[0]
print(f"  mp_id: {sample.mp_id}")
print(f"  atoms: {len(sample.atomic_numbers)}")
print(f"  edges: {sample.edge_index.shape[1]}")
