#!/usr/bin/env python3
"""Unit tests for enhanced graph backbone components."""

import numpy as np
import torch
import pytest

from mp_data_pipeline.models.graph_features import (
    compute_triplet_angles,
    AngleExpansion,
)
from mp_data_pipeline.models.enhanced_backbones import (
    EnhancedRBFExpansion,
    EnhancedGraphBackbone,
)


def test_compute_triplet_angles_simple():
    """Test angle computation on a simple triangle."""
    # Create a simple triangle: 3 atoms forming 90-degree angle
    positions = np.array([
        [0.0, 0.0, 0.0],  # atom 0
        [1.0, 0.0, 0.0],  # atom 1
        [1.0, 1.0, 0.0],  # atom 2
    ], dtype=np.float32)

    # Edges: 0->1, 1->0, 1->2, 2->1
    edge_index = np.array([
        [0, 1, 1, 2],
        [1, 0, 2, 1],
    ], dtype=np.int64)

    angles, triplet_index = compute_triplet_angles(edge_index, positions)

    # Should have 1 triplet: (0, 1, 2) with 90-degree angle
    assert len(angles) == 1
    assert triplet_index.shape == (3, 1)

    # Check angle is approximately 90 degrees (π/2 radians)
    expected_angle = np.pi / 2
    assert np.abs(angles[0] - expected_angle) < 0.01


def test_compute_triplet_angles_empty():
    """Test angle computation with no valid triplets."""
    positions = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    edge_index = np.array([[0], [0]], dtype=np.int64)

    angles, triplet_index = compute_triplet_angles(edge_index, positions)

    assert len(angles) == 0
    assert triplet_index.shape[1] == 0


def test_angle_expansion():
    """Test angle expansion using RBF."""
    expansion = AngleExpansion(n_angle_basis=32)

    # Test with a few angles
    angles = torch.tensor([0.0, np.pi/4, np.pi/2, np.pi], dtype=torch.float32)
    expanded = expansion(angles)

    assert expanded.shape == (4, 32)
    assert torch.all(expanded >= 0)  # RBF outputs are non-negative
    assert torch.all(expanded <= 1)  # RBF outputs are at most 1


def test_enhanced_rbf_expansion():
    """Test enhanced RBF expansion with more basis functions."""
    expansion = EnhancedRBFExpansion(n_rbf=128, cutoff=8.0)

    # Test with a few distances
    distances = torch.tensor([0.0, 2.0, 4.0, 6.0, 8.0], dtype=torch.float32)
    expanded = expansion(distances)

    assert expanded.shape == (5, 128)
    assert torch.all(expanded >= 0)
    assert torch.all(expanded <= 1)


def test_enhanced_graph_backbone_forward():
    """Test forward pass of EnhancedGraphBackbone."""
    backbone = EnhancedGraphBackbone(
        hidden_dim=64,
        n_layers=3,
        cutoff=8.0,
        n_rbf=128,
        use_angles=False,  # Test without angles first
        use_edge_update=False,
    )

    # Create a simple batch with 2 graphs
    batch_dict = {
        "z": torch.tensor([6, 8, 1, 6], dtype=torch.long),  # C, O, H, C
        "edge_index": torch.tensor([
            [0, 1, 2, 3],
            [1, 0, 3, 2],
        ], dtype=torch.long),
        "edge_dist": torch.tensor([1.5, 1.5, 1.1, 1.1], dtype=torch.float32),
        "batch": torch.tensor([0, 0, 1, 1], dtype=torch.long),
    }

    output = backbone(batch_dict)

    # Should output (2, 64) for 2 graphs
    assert output.shape == (2, 64)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def test_enhanced_graph_backbone_with_amp():
    """Test that EnhancedGraphBackbone works with AMP."""
    backbone = EnhancedGraphBackbone(
        hidden_dim=64,
        n_layers=3,
        cutoff=8.0,
        n_rbf=128,
    )
    backbone = backbone.cuda() if torch.cuda.is_available() else backbone

    batch_dict = {
        "z": torch.tensor([6, 8, 1, 6], dtype=torch.long),
        "edge_index": torch.tensor([
            [0, 1, 2, 3],
            [1, 0, 3, 2],
        ], dtype=torch.long),
        "edge_dist": torch.tensor([1.5, 1.5, 1.1, 1.1], dtype=torch.float32),
        "batch": torch.tensor([0, 0, 1, 1], dtype=torch.long),
    }

    if torch.cuda.is_available():
        batch_dict = {k: v.cuda() if isinstance(v, torch.Tensor) else v
                      for k, v in batch_dict.items()}

    # Test with AMP
    with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
        output = backbone(batch_dict)

    assert output.shape == (2, 64)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def test_enhanced_graph_backbone_backward():
    """Test backward pass (gradient computation)."""
    backbone = EnhancedGraphBackbone(
        hidden_dim=64,
        n_layers=3,
        cutoff=8.0,
        n_rbf=128,
    )

    batch_dict = {
        "z": torch.tensor([6, 8, 1, 6], dtype=torch.long),
        "edge_index": torch.tensor([
            [0, 1, 2, 3],
            [1, 0, 3, 2],
        ], dtype=torch.long),
        "edge_dist": torch.tensor([1.5, 1.5, 1.1, 1.1], dtype=torch.float32),
        "batch": torch.tensor([0, 0, 1, 1], dtype=torch.long),
    }

    output = backbone(batch_dict)
    loss = output.sum()
    loss.backward()

    # Check that gradients are computed
    for name, param in backbone.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
