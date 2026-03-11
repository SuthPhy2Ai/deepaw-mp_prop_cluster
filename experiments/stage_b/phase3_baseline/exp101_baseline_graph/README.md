# Experiment 101: Baseline Graph (Stage B)

**Date**: 2026-03-06
**Status**: 🔄 Running
**Phase**: 3 (Stage B)

---

## Overview

Stage B baseline experiment with 18 tasks including elastic properties. This is the first experiment to train on all tasks including elastic/mechanical properties.

## Configuration

### Changes from Stage A Baseline (exp001)

- **Tasks**: 8 → 18 tasks (added 10 tasks including elastic properties)
- **Sampling**: Added weighted sampling with `--oversample-elastic 4.0`
- **Architecture**: Same as exp001 (graph backbone, 6.0Å cutoff, 24 neighbors)

### Training Setup

- **Backbone**: graph (SchNet-style)
- **Stage**: b (all 18 tasks)
- **Batch size**: 64
- **Learning rate**: 2e-4
- **Epochs**: 100
- **Weighted sampling**: 4× oversampling for elastic data
- **Excluded tasks**: None (Stage B includes all tasks)

### Task List (18 tasks)

**Thermodynamic (3)**:
- energy_per_atom
- formation_energy_per_atom
- energy_above_hull

**Electronic (5)**:
- band_gap
- cbm
- vbm
- efermi
- is_metal

**Structural (2)**:
- volume
- density

**Stability (1)**:
- is_stable

**Elastic/Mechanical (7)**:
- bulk_modulus_vrh
- shear_modulus_vrh
- youngs_modulus
- homogeneous_poisson
- universal_anisotropy
- (2 more elastic properties)

## Expected Challenges

1. **Low elastic data coverage**: Only ~7% of materials have elastic data
2. **Task imbalance**: Need careful loss weighting
3. **Longer training time**: More tasks to learn

## Results

[To be filled after training]

## Key Findings

[To be filled after analysis]

## Files

- `model_checkpoint.pt` - Best model weights
- `config.json` - Full training configuration
- `training_log.txt` - Console output
- `tensorboard/` - Training curves
- `analysis/` - Post-training analysis

## Next Steps

[Future improvements or follow-up experiments]
