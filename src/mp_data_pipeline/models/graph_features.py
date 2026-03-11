#!/usr/bin/env python3
"""Graph feature engineering utilities for enhanced Graph Backbone.

This module provides functions for computing angle features (three-body interactions)
and encoding them using radial basis functions or spherical harmonics.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Optional


def compute_triplet_angles(
    edge_index: np.ndarray,
    positions: np.ndarray,
    batch: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute three-atom angles (i-j-k) where j is the center atom.

    For each edge (i, j), we find all other edges (k, j) sharing the same
    destination node j. The angle is computed between vectors v_ij and v_kj.

    Args:
        edge_index: (2, num_edges) array of [src, dst] indices
        positions: (num_nodes, 3) array of atomic positions
        batch: (num_nodes,) array of graph indices (optional, for batched graphs)

    Returns:
        angles: (num_triplets,) array of angles in radians [0, π]
        triplet_index: (3, num_triplets) array of [i, j, k] indices
    """
    src, dst = edge_index[0], edge_index[1]
    num_edges = len(src)

    # Group edges by destination node
    # For each node j, find all edges (*, j)
    dst_to_edges = {}
    for edge_idx in range(num_edges):
        j = dst[edge_idx]
        if j not in dst_to_edges:
            dst_to_edges[j] = []
        dst_to_edges[j].append(edge_idx)

    # Compute angles for all triplets
    angles_list = []
    triplets_list = []

    for j, edge_indices in dst_to_edges.items():
        if len(edge_indices) < 2:
            # Need at least 2 edges to form an angle
            continue

        # Get all pairs of edges sharing node j
        for idx1 in range(len(edge_indices)):
            for idx2 in range(idx1 + 1, len(edge_indices)):
                edge_idx1 = edge_indices[idx1]
                edge_idx2 = edge_indices[idx2]

                i = src[edge_idx1]
                k = src[edge_idx2]

                # Skip if i and k are in different graphs (for batched data)
                if batch is not None and batch[i] != batch[k]:
                    continue

                # Compute vectors v_ij and v_kj
                pos_i = positions[i]
                pos_j = positions[j]
                pos_k = positions[k]

                v_ij = pos_i - pos_j
                v_kj = pos_k - pos_j

                # Compute angle using dot product
                norm_ij = np.linalg.norm(v_ij)
                norm_kj = np.linalg.norm(v_kj)

                if norm_ij < 1e-8 or norm_kj < 1e-8:
                    # Skip degenerate cases
                    continue

                cos_angle = np.dot(v_ij, v_kj) / (norm_ij * norm_kj)
                # Clamp to [-1, 1] to avoid numerical issues with arccos
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle = np.arccos(cos_angle)

                angles_list.append(angle)
                triplets_list.append([i, j, k])

    if len(angles_list) == 0:
        # No valid triplets found
        return np.array([], dtype=np.float32), np.array([[], [], []], dtype=np.int64)

    angles = np.array(angles_list, dtype=np.float32)
    triplet_index = np.array(triplets_list, dtype=np.int64).T  # (3, num_triplets)

    return angles, triplet_index


def map_triplets_to_edges(
    triplet_index: np.ndarray,
    edge_index: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Map triplet angles to their corresponding edges.

    For each triplet (i, j, k), we have two edges: (i, j) and (k, j).
    This function finds the edge indices for these edges.

    Args:
        triplet_index: (3, num_triplets) array of [i, j, k] indices
        edge_index: (2, num_edges) array of [src, dst] indices

    Returns:
        edge1_idx: (num_triplets,) indices of edges (i, j)
        edge2_idx: (num_triplets,) indices of edges (k, j)
    """
    num_triplets = triplet_index.shape[1]
    edge1_idx = np.zeros(num_triplets, dtype=np.int64)
    edge2_idx = np.zeros(num_triplets, dtype=np.int64)

    # Create edge lookup dict for fast search
    edge_dict = {}
    for edge_idx in range(edge_index.shape[1]):
        src, dst = edge_index[0, edge_idx], edge_index[1, edge_idx]
        edge_dict[(src, dst)] = edge_idx

    for t in range(num_triplets):
        i, j, k = triplet_index[:, t]
        edge1_idx[t] = edge_dict.get((i, j), -1)
        edge2_idx[t] = edge_dict.get((k, j), -1)

    return edge1_idx, edge2_idx


class AngleExpansion(nn.Module):
    """Expand angles using Gaussian radial basis functions.

    Similar to RBFExpansion but for angles in [0, π] range.
    """

    def __init__(self, n_angle_basis: int = 32, trainable: bool = False):
        """Initialize angle expansion.

        Args:
            n_angle_basis: Number of basis functions
            trainable: Whether basis centers and widths are trainable
        """
        super().__init__()
        self.n_angle_basis = n_angle_basis

        # Centers uniformly distributed in [0, π]
        centers = torch.linspace(0, np.pi, n_angle_basis)
        self.register_buffer("centers", centers)

        # Width parameter (gamma = 1 / (2 * sigma^2))
        # Use spacing between centers to set width
        spacing = np.pi / (n_angle_basis - 1) if n_angle_basis > 1 else 1.0
        gamma = 1.0 / (2 * spacing ** 2)

        if trainable:
            self.centers = nn.Parameter(self.centers)
            self.gamma = nn.Parameter(torch.tensor(gamma))
        else:
            self.register_buffer("gamma", torch.tensor(gamma))

    def forward(self, angles: torch.Tensor) -> torch.Tensor:
        """Expand angles using Gaussian RBF.

        Args:
            angles: (num_triplets,) tensor of angles in radians

        Returns:
            (num_triplets, n_angle_basis) tensor of expanded features
        """
        # angles: (num_triplets,)
        # centers: (n_angle_basis,)
        # output: (num_triplets, n_angle_basis)

        diff = angles.unsqueeze(-1) - self.centers  # (num_triplets, n_angle_basis)
        return torch.exp(-self.gamma * diff * diff)
