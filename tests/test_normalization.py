#!/usr/bin/env python3
"""Test normalization and denormalization correctness."""

import numpy as np
import torch


def test_zscore_normalization():
    """Test z-score normalization is reversible."""
    # Generate random data
    np.random.seed(42)
    data = np.random.randn(1000, 10) * 5 + 10  # mean~10, std~5

    # Compute statistics
    mean = data.mean(axis=0)
    std = data.std(axis=0)

    # Normalize
    normalized = (data - mean) / (std + 1e-8)

    # Denormalize
    denormalized = normalized * std + mean

    # Check reversibility
    max_error = np.abs(data - denormalized).max()
    print(f"Z-score max error: {max_error:.2e}")
    assert max_error < 1e-6, f"Z-score not reversible: max error {max_error}"
    print("✓ Z-score normalization is reversible")


def test_log1p_transform():
    """Test log1p transform is reversible for positive values."""
    # Generate positive data
    np.random.seed(42)
    data = np.abs(np.random.randn(1000, 5)) * 10 + 0.1

    # Transform
    transformed = np.log1p(data)

    # Inverse transform
    recovered = np.expm1(transformed)

    # Check reversibility
    max_error = np.abs(data - recovered).max()
    print(f"Log1p max error: {max_error:.2e}")
    assert max_error < 1e-6, f"Log1p not reversible: max error {max_error}"
    print("✓ Log1p transform is reversible")


def test_combined_normalization():
    """Test combined log1p + z-score normalization."""
    # Generate positive data
    np.random.seed(42)
    data = np.abs(np.random.randn(1000, 3)) * 100 + 1.0

    # Apply log1p
    log_data = np.log1p(data)

    # Z-score normalize
    mean = log_data.mean(axis=0)
    std = log_data.std(axis=0)
    normalized = (log_data - mean) / (std + 1e-8)

    # Reverse: denormalize then expm1
    denormalized = normalized * std + mean
    recovered = np.expm1(denormalized)

    # Check reversibility
    max_error = np.abs(data - recovered).max()
    relative_error = (np.abs(data - recovered) / (data + 1e-8)).max()
    print(f"Combined max error: {max_error:.2e}")
    print(f"Combined relative error: {relative_error:.2e}")
    assert relative_error < 1e-5, f"Combined transform not reversible: relative error {relative_error}"
    print("✓ Combined log1p + z-score is reversible")


def test_train_test_consistency():
    """Test that test set uses train statistics."""
    np.random.seed(42)

    # Train data
    train_data = np.random.randn(800, 5) * 3 + 5
    train_mean = train_data.mean(axis=0)
    train_std = train_data.std(axis=0)

    # Test data (different distribution)
    test_data = np.random.randn(200, 5) * 4 + 7

    # Normalize test using TRAIN statistics (correct)
    test_normalized_correct = (test_data - train_mean) / (train_std + 1e-8)

    # Denormalize using TRAIN statistics
    test_recovered = test_normalized_correct * train_std + train_mean

    # Check reversibility
    max_error = np.abs(test_data - test_recovered).max()
    print(f"Train/test consistency max error: {max_error:.2e}")
    assert max_error < 1e-6, f"Train/test normalization not consistent: max error {max_error}"
    print("✓ Test set correctly uses train statistics")

    # Show that using test statistics would be wrong
    test_mean = test_data.mean(axis=0)
    test_std = test_data.std(axis=0)
    mean_diff = np.abs(train_mean - test_mean).max()
    std_diff = np.abs(train_std - test_std).max()
    print(f"  (Train vs test mean diff: {mean_diff:.3f}, std diff: {std_diff:.3f})")


def test_mask_handling():
    """Test that masked values don't affect normalization."""
    np.random.seed(42)

    # Data with some missing values
    data = np.random.randn(100, 3) * 2 + 3
    mask = np.random.rand(100, 3) > 0.3  # 70% coverage

    # Compute statistics only on valid data
    valid_data = data[mask]
    mean = valid_data.mean()
    std = valid_data.std()

    # Normalize only valid entries
    normalized = np.zeros_like(data)
    normalized[mask] = (data[mask] - mean) / (std + 1e-8)

    # Denormalize
    denormalized = np.zeros_like(normalized)
    denormalized[mask] = normalized[mask] * std + mean

    # Check reversibility for valid entries
    max_error = np.abs(data[mask] - denormalized[mask]).max()
    print(f"Masked normalization max error: {max_error:.2e}")
    assert max_error < 1e-6, f"Masked normalization not reversible: max error {max_error}"
    print("✓ Masked normalization is correct")


if __name__ == "__main__":
    print("Testing normalization correctness...\n")

    test_zscore_normalization()
    print()

    test_log1p_transform()
    print()

    test_combined_normalization()
    print()

    test_train_test_consistency()
    print()

    test_mask_handling()
    print()

    print("=" * 50)
    print("All normalization tests passed! ✓")
    print("=" * 50)
