"""PyG InMemoryDataset for fast training with zero I/O overhead."""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Sequence

import numpy as np
import torch
from ase.db import connect
from ase.neighborlist import neighbor_list
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm

from .tasks import ELASTIC_TASKS, TASK_NAME_LIST


class PyGMaterialsDataset(InMemoryDataset):
    """PyG InMemoryDataset for materials property prediction.

    All data is loaded into memory at initialization for maximum speed.
    Eliminates SQLite I/O bottleneck completely.
    """

    def __init__(
        self,
        root: str,
        db_path: str,
        mp_ids: Sequence[str],
        cutoff: float = 6.0,
        max_neighbors: int = 24,
        strict_elastic_filter: bool = True,
        transform=None,
        pre_transform=None,
    ):
        self.db_path = Path(db_path)
        self.mp_ids = list(mp_ids)
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
        self.strict_elastic_filter = strict_elastic_filter

        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return []

    @property
    def processed_file_names(self) -> List[str]:
        # Include config in filename to avoid cache conflicts
        return [f'data_c{self.cutoff}_n{self.max_neighbors}.pt']

    def process(self):
        """Convert ASE database to PyG Data objects and save to disk."""
        print(f"Processing {len(self.mp_ids)} materials from ASE database...")
        print(f"This is a one-time conversion. Subsequent runs will load from cache.")

        # Load graph cache first (much faster than recomputing)
        graph_cache = self._load_graph_cache()
        if not graph_cache:
            print("⚠️ Warning: No graph cache found. This will be very slow.")
            print("   Consider running: python scripts/precompute_graphs.py")

        db = connect(str(self.db_path))
        data_list = []

        for mp_id in tqdm(self.mp_ids, desc="Converting to PyG"):
            try:
                row = db.get(mp_id=mp_id)
            except KeyError:
                print(f"Warning: mp_id {mp_id} not found, skipping")
                continue
            except Exception as e:
                print(f"Warning: mp_id {mp_id} corrupted ({str(e)[:50]}), skipping")
                continue

            # Get atoms structure
            try:
                atoms = row.toatoms()
            except Exception as e:
                print(f"Warning: mp_id {mp_id} failed to convert to atoms ({str(e)[:50]}), skipping")
                continue
            atomic_numbers = torch.tensor(atoms.numbers, dtype=torch.long)
            positions = torch.tensor(atoms.positions, dtype=torch.float32)

            # Use cached graph if available, otherwise compute
            if graph_cache and mp_id in graph_cache:
                edge_index_np, edge_dist_np = graph_cache[mp_id]
                edge_index = torch.tensor(edge_index_np, dtype=torch.long)
                edge_attr = torch.tensor(edge_dist_np, dtype=torch.float32).unsqueeze(1)
            else:
                # Fallback: compute graph on-the-fly (slow)
                src, dst, dist = neighbor_list("ijd", atoms, self.cutoff)

                if len(src) == 0:
                    n = len(atomic_numbers)
                    src = np.arange(n, dtype=np.int64)
                    dst = np.arange(n, dtype=np.int64)
                    dist = np.zeros(n, dtype=np.float32)

                if self.max_neighbors > 0:
                    src, dst, dist = self._limit_neighbors(src, dst, dist, len(atomic_numbers))

                edge_index = torch.tensor(np.vstack([src, dst]), dtype=torch.long)
                edge_attr = torch.tensor(dist, dtype=torch.float32).unsqueeze(1)

            # Extract targets and masks (fast)
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

            # Create PyG Data object
            data = Data(
                x=atomic_numbers,
                pos=positions,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=torch.tensor(targets, dtype=torch.float32),
                mask=torch.tensor(masks, dtype=torch.float32),
                mp_id=mp_id,
            )

            data_list.append(data)

        print(f"✅ Converted {len(data_list)} materials")
        print(f"Saving to {self.processed_paths[0]}...")

        # Save to disk
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

        print(f"✅ Saved PyG dataset cache")

    def _load_graph_cache(self) -> dict:
        """Load precomputed graph cache."""
        import hashlib
        import pickle

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
            return {}

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
            return 0 < value < 1000
        if task_name == "homogeneous_poisson":
            return -1.0 <= value <= 0.5
        if task_name == "universal_anisotropy":
            return 0 <= value < 100
        return True

    def _limit_neighbors(
        self,
        src: np.ndarray,
        dst: np.ndarray,
        dist: np.ndarray,
        n_nodes: int,
    ) -> tuple:
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

        return src[selected], dst[selected], dist[selected]


def collate_pyg_batch(batch):
    """Collate function for PyG Data objects.

    Converts PyG Batch to the dict format expected by the model.
    """
    from torch_geometric.data import Batch

    pyg_batch = Batch.from_data_list(batch)

    # PyG flattens y and mask, need to reshape them back to [batch_size, num_tasks]
    batch_size = len(batch)
    num_tasks = batch[0].y.shape[0]

    # Convert to dict format expected by model
    batch_dict = {
        'z': pyg_batch.x,  # atomic numbers
        'pos': pyg_batch.pos,  # positions
        'edge_index': pyg_batch.edge_index,
        'edge_dist': pyg_batch.edge_attr.squeeze(-1),
        'batch': pyg_batch.batch,
        'targets': pyg_batch.y.view(batch_size, num_tasks),
        'masks': pyg_batch.mask.view(batch_size, num_tasks),
    }

    return batch_dict
