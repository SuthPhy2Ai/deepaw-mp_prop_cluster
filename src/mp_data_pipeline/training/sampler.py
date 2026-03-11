"""Sampler utilities for balancing low-coverage tasks."""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import List

import numpy as np
from torch.utils.data import WeightedRandomSampler

from ..ml.tasks import ELASTIC_TASKS, TASK_INDEX


def build_stage_sampler(dataset, stage: str, oversample_elastic: float = 4.0, split_file: str = None):
    """Build a weighted sampler for stage B/C to boost elastic-label samples.

    Uses caching to avoid recomputing weights on every run.
    """
    stage_lower = stage.lower()
    if stage_lower not in {"b", "c", "full"}:
        return None

    # Try to load cached weights
    cache_key = hashlib.md5(
        f"{split_file}_{dataset.cutoff}_{dataset.max_neighbors}_{oversample_elastic}".encode()
    ).hexdigest()
    cache_dir = Path(__file__).resolve().parents[3] / "data" / "cache"
    cache_dir.mkdir(exist_ok=True, parents=True)
    cache_file = cache_dir / f"sampler_weights_{cache_key}.pkl"

    if cache_file.exists():
        print(f"Loading cached sampler weights from: {cache_file}")
        try:
            with open(cache_file, 'rb') as f:
                weights = pickle.load(f)
            elastic_count = int((weights > 1.0).sum())
            print(f"✅ Loaded weights for {len(weights)} samples")
            print(f"   {elastic_count} samples with elastic data ({100*elastic_count/len(weights):.2f}%)")
            return WeightedRandomSampler(weights.tolist(), num_samples=len(weights), replacement=True)
        except Exception as e:
            print(f"⚠️ Failed to load cache: {e}, recomputing...")

    # Compute weights
    elastic_indices: List[int] = [TASK_INDEX[t] for t in ELASTIC_TASKS]
    weights = np.ones(len(dataset), dtype=np.float32)

    print(f"Building weighted sampler for {len(dataset)} samples...")
    print(f"Elastic tasks: {ELASTIC_TASKS}")
    print(f"Oversample factor: {oversample_elastic}x")
    print(f"This will take ~{len(dataset)/44/60:.1f} minutes at 44 samples/sec...")

    # Iterate through dataset to check which samples have elastic data
    for i in range(len(dataset)):
        if i % 10000 == 0 and i > 0:
            print(f"  Processed {i}/{len(dataset)} samples ({100*i/len(dataset):.1f}%)...")

        sample = dataset[i]
        has_elastic = any(sample.masks[idx] > 0.5 for idx in elastic_indices)
        if has_elastic:
            weights[i] = float(oversample_elastic)

    elastic_count = int((weights > 1.0).sum())
    print(f"✅ Found {elastic_count} samples with elastic data ({100*elastic_count/len(dataset):.2f}%)")
    print(f"   Will oversample these by {oversample_elastic}x")

    # Save cache
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(weights, f)
        print(f"💾 Saved weights cache to {cache_file}")
    except Exception as e:
        print(f"⚠️ Failed to save cache: {e}")

    return WeightedRandomSampler(weights.tolist(), num_samples=len(weights), replacement=True)
