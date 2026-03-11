"""Enhanced ASE database dataset with angle feature computation.

This module extends AseGraphMultitaskDataset to compute angle features
(three-body interactions) during graph construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from ase.db import connect
from ase.neighborlist import neighbor_list
from torch.utils.data import Dataset

from .dataset import AseGraphMultitaskDataset, GraphSample
from .tasks import ELASTIC_TASKS, TASK_NAME_LIST
from ..models.graph_features import compute_triplet_angles


@dataclass
class EnhancedGraphSample:
    """Enhanced graph sample with angle features."""

    mp_id: str
    atomic_numbers: np.ndarray
    edge_index: np.ndarray
    edge_dist: np.ndarray
    targets: np.ndarray
    masks: np.ndarray
    # New fields for angle features
    edge_angles: np.ndarray  # (num_triplets,) angles in radians
    triplet_index: np.ndarray  # (3, num_triplets) [i, j, k] indices
    # New field for DeePAW features
    positions: np.ndarray  # (num_atoms, 3) atomic positions


class EnhancedGraphDataset(AseGraphMultitaskDataset):
    """Enhanced multitask graph dataset with angle feature computation.

    This dataset extends AseGraphMultitaskDataset to compute angle features
    during graph construction. It supports larger graph coverage and optional
    angle computation.
    """

    def __init__(
        self,
        db_path: Path,
        mp_ids: Sequence[str],
        cutoff: float = 8.0,  # Increased from 6.0
        max_neighbors: int = 48,  # Increased from 24
        strict_elastic_filter: bool = True,
        compute_angles: bool = True,  # New parameter
    ):
        """Initialize enhanced graph dataset.

        Args:
            db_path: Path to ASE database
            mp_ids: List of Materials Project IDs to load
            cutoff: Cutoff radius for graph construction (Angstroms)
            max_neighbors: Maximum number of neighbors per atom
            strict_elastic_filter: Whether to filter invalid elastic values
            compute_angles: Whether to compute angle features
        """
        self.compute_angles = compute_angles
        # Call parent constructor
        super().__init__(
            db_path=db_path,
            mp_ids=mp_ids,
            cutoff=cutoff,
            max_neighbors=max_neighbors,
            strict_elastic_filter=strict_elastic_filter,
        )

    def _row_to_sample(self, row) -> EnhancedGraphSample:
        """Convert ASE database row to enhanced graph sample.

        Args:
            row: ASE database row

        Returns:
            EnhancedGraphSample with graph structure and angle features
        """
        mp_id = row.get("mp_id") or f"id-{row.id}"
        atoms = row.toatoms()
        atomic_numbers = np.asarray(atoms.numbers, dtype=np.int64)
        positions = atoms.get_positions()  # (num_atoms, 3)

        # Build graph structure
        src, dst, dist = neighbor_list("ijd", atoms, self.cutoff)

        if len(src) == 0:
            # Fallback for very sparse edge cases
            n = len(atomic_numbers)
            src = np.arange(n, dtype=np.int64)
            dst = np.arange(n, dtype=np.int64)
            dist = np.zeros(n, dtype=np.float32)

        if self.max_neighbors > 0:
            src, dst, dist = self._limit_neighbors(src, dst, dist, len(atomic_numbers))

        edge_index = np.vstack([src, dst]).astype(np.int64)
        edge_dist = np.asarray(dist, dtype=np.float32)

        # Compute angle features if requested
        if self.compute_angles:
            edge_angles, triplet_index = compute_triplet_angles(
                edge_index=edge_index,
                positions=positions,
                batch=None,  # Single graph, no batch
            )
        else:
            # Empty arrays if angles not computed
            edge_angles = np.array([], dtype=np.float32)
            triplet_index = np.array([[], [], []], dtype=np.int64)

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

            # Clip to float32 range to avoid overflow
            val_f = np.clip(val_f, -3.4e38, 3.4e38)
            targets[idx] = val_f
            masks[idx] = 1.0

        return EnhancedGraphSample(
            mp_id=mp_id,
            atomic_numbers=atomic_numbers,
            edge_index=edge_index,
            edge_dist=edge_dist,
            targets=targets,
            masks=masks,
            edge_angles=edge_angles,
            triplet_index=triplet_index,
            positions=positions.astype(np.float32),
        )


def collate_enhanced_graph_samples(
    samples: Sequence[EnhancedGraphSample],
) -> Dict[str, torch.Tensor]:
    """Collate enhanced graph samples into a batched graph dictionary.

    Args:
        samples: List of EnhancedGraphSample objects

    Returns:
        Dictionary containing batched graph data with angle features
    """
    z_list = []
    edge_index_list = []
    edge_dist_list = []
    batch_index_list = []
    y_list = []
    mask_list = []
    mp_ids = []

    # Angle features
    edge_angles_list = []
    triplet_index_list = []

    # DeePAW features
    positions_list = []

    node_offset = 0
    for batch_idx, sample in enumerate(samples):
        n_nodes = sample.atomic_numbers.shape[0]

        z_list.append(torch.from_numpy(sample.atomic_numbers))
        edge_index = torch.from_numpy(sample.edge_index)
        edge_index_list.append(edge_index + node_offset)
        edge_dist_list.append(torch.from_numpy(sample.edge_dist))
        batch_index_list.append(torch.full((n_nodes,), batch_idx, dtype=torch.long))

        y_list.append(torch.from_numpy(sample.targets))
        mask_list.append(torch.from_numpy(sample.masks))
        mp_ids.append(sample.mp_id)

        # Add angle features
        if len(sample.edge_angles) > 0:
            edge_angles_list.append(torch.from_numpy(sample.edge_angles))
            triplet_index = torch.from_numpy(sample.triplet_index)
            # Offset triplet indices by node_offset
            triplet_index_list.append(triplet_index + node_offset)

        # Add positions
        positions_list.append(torch.from_numpy(sample.positions))

        node_offset += n_nodes

    batch_dict = {
        "z": torch.cat(z_list, dim=0).long(),
        "edge_index": torch.cat(edge_index_list, dim=1).long(),
        "edge_dist": torch.cat(edge_dist_list, dim=0).float(),
        "batch": torch.cat(batch_index_list, dim=0).long(),
        "targets": torch.stack(y_list, dim=0).float(),
        "masks": torch.stack(mask_list, dim=0).float(),
        "mp_ids": mp_ids,
        "pos": torch.cat(positions_list, dim=0).float(),  # (total_nodes, 3)
    }

    # Add angle features if available
    if edge_angles_list:
        batch_dict["edge_angles"] = torch.cat(edge_angles_list, dim=0).float()
        batch_dict["triplet_index"] = torch.cat(triplet_index_list, dim=1).long()

    return batch_dict
