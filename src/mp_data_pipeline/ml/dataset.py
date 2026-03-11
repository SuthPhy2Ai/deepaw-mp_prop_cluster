"""ASE database dataset and graph featurization for multitask training."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from ase.db import connect
from ase.neighborlist import neighbor_list
from torch.utils.data import Dataset

from .tasks import ELASTIC_TASKS, TASK_NAME_LIST


@dataclass
class GraphSample:
    """Single graph sample used by the dataloader."""

    mp_id: str
    atomic_numbers: np.ndarray
    positions: np.ndarray  # Atomic positions (N, 3)
    edge_index: np.ndarray
    edge_dist: np.ndarray
    targets: np.ndarray
    masks: np.ndarray


class AseGraphMultitaskDataset(Dataset):
    """Multitask graph dataset built from ASE DB rows.

    Uses lazy loading: queries database on-demand per sample instead of
    pre-loading all samples. This dramatically speeds up initialization.
    """

    def __init__(
        self,
        db_path: Path,
        mp_ids: Sequence[str],
        cutoff: float = 6.0,
        max_neighbors: int = 24,
        strict_elastic_filter: bool = True,
        preload_targets: bool = True,
    ):
        self.db_path = Path(db_path)
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
        self.strict_elastic_filter = strict_elastic_filter
        self.mp_ids = list(mp_ids)
        # Keep db connection open for lazy loading
        self.db = connect(str(self.db_path))

        # Try to load graph cache
        self.graph_cache = self._load_graph_cache()

        # Preload all targets and masks to memory (eliminates SQLite I/O bottleneck)
        self.preload_targets = preload_targets
        self.targets_cache = {}
        self.masks_cache = {}
        if preload_targets:
            self._preload_targets_and_masks()

        # Verify all mp_ids exist (quick check without loading data)
        self._verify_mp_ids()

    def _load_graph_cache(self) -> dict:
        """Try to load precomputed graph cache."""
        import hashlib
        import pickle

        # Create cache key based on dataset configuration
        # Use absolute path to match precompute_graphs.py
        abs_db_path = self.db_path.resolve()
        cache_key = hashlib.md5(
            f"{abs_db_path}_{self.cutoff}_{self.max_neighbors}".encode()
        ).hexdigest()
        cache_file = Path(__file__).resolve().parents[3] / "data" / "cache" / f"graphs_{cache_key}.pkl"
        
        if cache_file.exists():
            print(f"Loading graph cache from: {cache_file}")
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                print(f"✅ Loaded {len(cached_data['graphs'])} cached graphs")
                return cached_data['graphs']
            except Exception as e:
                print(f"⚠️ Failed to load graph cache: {e}")
                return {}
        else:
            print(f"⚠️ No graph cache found at {cache_file}")
            print(f"   Run: python scripts/precompute_graphs.py --cutoff {self.cutoff} --max-neighbors {self.max_neighbors}")
            return {}

    def _preload_targets_and_masks(self) -> None:
        """Preload all targets and masks to memory to eliminate SQLite I/O bottleneck."""
        print(f"Preloading targets and masks for {len(self.mp_ids)} samples...")
        from tqdm import tqdm

        for mp_id in tqdm(self.mp_ids, desc="Preloading", leave=False):
            try:
                row = self.db.get(mp_id=mp_id)
            except KeyError:
                continue

            # Extract targets and masks
            targets = np.zeros(len(TASK_NAME_LIST), dtype=np.float32)
            masks = np.zeros(len(TASK_NAME_LIST), dtype=np.float32)

            for idx, task_name in enumerate(TASK_NAME_LIST):
                val = row.get(task_name)
                if not self._is_valid_value(val):
                    continue

                val_f = float(val)
                if task_name in ELASTIC_TASKS and self.strict_elastic_filter:
                    if not self._valid_elastic(task_name, val_f):
                        continue

                # Filter extreme structural values
                if task_name == "volume" and (val_f <= 0 or val_f > 10000):
                    continue
                if task_name == "density" and (val_f <= 0 or val_f > 50):
                    continue

                targets[idx] = val_f
                masks[idx] = 1.0

            self.targets_cache[mp_id] = targets
            self.masks_cache[mp_id] = masks

        print(f"✅ Preloaded {len(self.targets_cache)} samples to memory (~{len(self.targets_cache) * 144 / 1024 / 1024:.1f} MB)")

    def _verify_mp_ids(self) -> None:
        """Quick verification that all mp_ids exist in database."""
        # Sample check: verify first 10 and last 10 mp_ids
        check_ids = self.mp_ids[:10] + self.mp_ids[-10:]
        missing = []
        for mp_id in check_ids:
            try:
                self.db.get(mp_id=mp_id)
            except KeyError:
                missing.append(mp_id)

        if missing:
            raise ValueError(f"Missing mp_ids from DB: {missing}")

    def __len__(self) -> int:
        return len(self.mp_ids)

    def __getitem__(self, idx: int) -> GraphSample:
        """Lazy load: query database and build graph on-demand."""
        mp_id = self.mp_ids[idx]
        try:
            row = self.db.get(mp_id=mp_id)
        except KeyError:
            raise ValueError(f"mp_id not found in database: {mp_id}")

        return self._row_to_sample(row)

    def get_masks(self, idx: int) -> np.ndarray:
        """Get only the masks for a sample (fast, no graph computation)."""
        mp_id = self.mp_ids[idx]
        try:
            row = self.db.get(mp_id=mp_id)
        except KeyError:
            raise ValueError(f"mp_id not found in database: {mp_id}")

        masks = np.zeros(len(TASK_NAME_LIST), dtype=np.float32)
        for task_idx, task_name in enumerate(TASK_NAME_LIST):
            val = row.get(task_name)
            if not self._is_valid_value(val):
                continue

            val_f = float(val)
            if task_name in ELASTIC_TASKS and self.strict_elastic_filter:
                if not self._valid_elastic(task_name, val_f):
                    continue

            # Filter extreme structural values (same as _row_to_sample)
            if task_name == "volume" and (val_f <= 0 or val_f > 10000):
                continue
            if task_name == "density" and (val_f <= 0 or val_f > 50):
                continue

            masks[task_idx] = 1.0

        return masks

    def _row_to_sample(self, row) -> GraphSample:
        mp_id = row.get("mp_id") or f"id-{row.id}"
        atoms = row.toatoms()
        atomic_numbers = np.asarray(atoms.numbers, dtype=np.int64)
        positions = np.asarray(atoms.positions, dtype=np.float32)

        # Try to use cached graph first
        if self.graph_cache and mp_id in self.graph_cache:
            edge_index, edge_dist = self.graph_cache[mp_id]
        else:
            # Fallback to on-the-fly computation
            src, dst, dist = neighbor_list("ijd", atoms, self.cutoff)

            if len(src) == 0:
                # Fallback for very sparse edge cases.
                n = len(atomic_numbers)
                src = np.arange(n, dtype=np.int64)
                dst = np.arange(n, dtype=np.int64)
                dist = np.zeros(n, dtype=np.float32)

            if self.max_neighbors > 0:
                src, dst, dist = self._limit_neighbors(src, dst, dist, len(atomic_numbers))

            edge_index = np.vstack([src, dst]).astype(np.int64)
            edge_dist = np.asarray(dist, dtype=np.float32)

        # Use preloaded targets/masks if available
        if self.preload_targets and mp_id in self.targets_cache:
            targets = self.targets_cache[mp_id]
            masks = self.masks_cache[mp_id]
        else:
            # Fallback to on-the-fly extraction
            targets = np.zeros(len(TASK_NAME_LIST), dtype=np.float32)
            masks = np.zeros(len(TASK_NAME_LIST), dtype=np.float32)

            for idx, task_name in enumerate(TASK_NAME_LIST):
                val = row.get(task_name)
                if not self._is_valid_value(val):
                    continue

                val_f = float(val)
                if task_name in ELASTIC_TASKS and self.strict_elastic_filter:
                    if not self._valid_elastic(task_name, val_f):
                        continue

                # Filter extreme structural values
                if task_name == "volume" and (val_f <= 0 or val_f > 10000):
                    continue
                if task_name == "density" and (val_f <= 0 or val_f > 50):
                    continue

                targets[idx] = val_f
                masks[idx] = 1.0

        return GraphSample(
            mp_id=mp_id,
            atomic_numbers=atomic_numbers,
            positions=positions,
            edge_index=edge_index,
            edge_dist=edge_dist,
            targets=targets,
            masks=masks,
        )

    @staticmethod
    def _is_valid_value(value) -> bool:
        if value is None:
            return False
        try:
            v = float(value)
        except (TypeError, ValueError):
            return False
        return not (math.isnan(v) or math.isinf(v))

    @staticmethod
    def _valid_elastic(task_name: str, value: float) -> bool:
        if task_name in {"bulk_modulus_vrh", "shear_modulus_vrh"}:
            # Reasonable range for elastic moduli in GPa (0-1000)
            return 0 < value < 1000
        if task_name == "homogeneous_poisson":
            return -1.0 <= value <= 0.5
        if task_name == "universal_anisotropy":
            # Reasonable range for anisotropy (0-100)
            return 0 <= value < 100
        return True

    def _limit_neighbors(
        self,
        src: np.ndarray,
        dst: np.ndarray,
        dist: np.ndarray,
        n_nodes: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(src) == 0:
            return src, dst, dist

        selected = []
        for node in range(n_nodes):
            node_idx = np.where(src == node)[0]
            if len(node_idx) <= self.max_neighbors:
                selected.extend(node_idx.tolist())
                continue
            ordered = node_idx[np.argsort(dist[node_idx])[: self.max_neighbors]]
            selected.extend(ordered.tolist())

        selected = np.asarray(selected, dtype=np.int64)
        return src[selected], dst[selected], dist[selected]


def collate_graph_samples(samples: Sequence[GraphSample]) -> Dict[str, torch.Tensor]:
    """Collate graph samples into a batched graph dictionary."""
    z_list = []
    pos_list = []
    edge_index_list = []
    edge_dist_list = []
    batch_index_list = []
    y_list = []
    mask_list = []
    mp_ids = []

    node_offset = 0
    for batch_idx, sample in enumerate(samples):
        n_nodes = sample.atomic_numbers.shape[0]

        z_list.append(torch.from_numpy(sample.atomic_numbers))
        pos_list.append(torch.from_numpy(sample.positions))
        edge_index = torch.from_numpy(sample.edge_index)
        edge_index_list.append(edge_index + node_offset)
        edge_dist_list.append(torch.from_numpy(sample.edge_dist))
        batch_index_list.append(torch.full((n_nodes,), batch_idx, dtype=torch.long))

        y_list.append(torch.from_numpy(sample.targets))
        mask_list.append(torch.from_numpy(sample.masks))
        mp_ids.append(sample.mp_id)

        node_offset += n_nodes

    return {
        "z": torch.cat(z_list, dim=0).long(),
        "pos": torch.cat(pos_list, dim=0).float(),
        "edge_index": torch.cat(edge_index_list, dim=1).long(),
        "edge_dist": torch.cat(edge_dist_list, dim=0).float(),
        "batch": torch.cat(batch_index_list, dim=0).long(),
        "targets": torch.stack(y_list, dim=0).float(),
        "masks": torch.stack(mask_list, dim=0).float(),
        "mp_ids": mp_ids,
    }
