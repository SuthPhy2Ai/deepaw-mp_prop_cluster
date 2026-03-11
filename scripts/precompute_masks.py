#!/usr/bin/env python
"""Precompute task masks for faster sampler initialization."""

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mp_data_pipeline.ml.dataset import AseGraphMultitaskDataset


def compute_masks(db_path: Path, mp_ids: list, cutoff: float, max_neighbors: int):
    """Compute mask matrix for all samples."""
    print(f"Computing masks for {len(mp_ids)} samples...")

    dataset = AseGraphMultitaskDataset(
        db_path=db_path,
        mp_ids=mp_ids,
        cutoff=cutoff,
        max_neighbors=max_neighbors,
    )

    masks = []
    for i in tqdm(range(len(dataset)), desc="Computing masks"):
        sample = dataset[i]
        masks.append(sample.masks)

    return np.array(masks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True, help="Split JSON file")
    parser.add_argument("--db", default="data/db/mp_materials.db")
    parser.add_argument("--cutoff", type=float, default=6.0)
    parser.add_argument("--max-neighbors", type=int, default=24)
    args = parser.parse_args()

    # Load split
    with open(args.split) as f:
        split_data = json.load(f)

    split = split_data.get("mp_ids", split_data)
    train_ids = split["train"]

    # Compute masks
    masks = compute_masks(Path(args.db), train_ids, args.cutoff, args.max_neighbors)

    # Save cache
    cache_key = hashlib.md5(
        f"{args.split}_{args.cutoff}_{args.max_neighbors}".encode()
    ).hexdigest()
    cache_dir = Path("data/cache")
    cache_dir.mkdir(exist_ok=True, parents=True)
    cache_file = cache_dir / f"masks_{cache_key}.pkl"

    with open(cache_file, 'wb') as f:
        pickle.dump({
            'masks': masks,
            'mp_ids': train_ids,
            'split': args.split,
            'cutoff': args.cutoff,
            'max_neighbors': args.max_neighbors,
        }, f)

    print(f"✅ Saved mask cache to {cache_file}")
    print(f"   Shape: {masks.shape}")
    print(f"   Size: {cache_file.stat().st_size / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()
