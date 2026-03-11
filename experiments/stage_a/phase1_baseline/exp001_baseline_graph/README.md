# Experiment 001: Baseline Graph Model

**Date**: 2026-03-05 to 2026-03-06
**Status**: ✅ Completed
**Phase**: 1

---

## Overview

Baseline graph neural network model using SchNet-style message passing for multitask crystal property prediction.

## Configuration

### Model Architecture
- **Backbone**: Graph (SchNet-style)
- **Hidden Dimension**: 256
- **Layers**: 6
- **Cutoff**: 6.0 Å
- **Max Neighbors**: 24
- **RBF Basis**: 64

### Training Setup
- **Learning Rate**: 0.0001
- **Batch Size**: 64
- **Epochs**: 50 (best at epoch 38)
- **Optimizer**: AdamW
- **Weight Decay**: 1e-5
- **Gradient Clipping**: 1.0
- **Warmup Epochs**: 5

### Data
- **Split**: IID (80/10/10)
- **Stage**: A (8 core tasks)
- **Excluded Tasks**: volume, density, is_stable
- **Training Samples**: 123,903
- **Validation Samples**: 15,487
- **Test Samples**: 15,489

## Results

### Overall Performance

| Split | Loss |
|-------|------|
| Train | 0.1454 |
| Val   | 0.2226 |
| Test  | 0.2223 |

### Key Metrics (Validation Set)

**Best Tasks**:
- formation_energy_per_atom: MAE = 0.0800 eV/atom, R² = 0.988
- energy_above_hull: MAE = 0.0644 eV/atom, R² = 0.924
- is_metal: AUROC = 0.9575

**Needs Improvement**:
- efermi: MAE = 0.3834 eV
- energy_per_atom: MAE = 0.3606 eV/atom
- cbm: MAE = 0.2921 eV

### Overfitting Analysis

Significant train-val gap observed:
- energy_per_atom: +95% (0.1848 → 0.3606)
- cbm: +57% (0.1861 → 0.2921)
- band_gap: +61% (0.1433 → 0.2308)

## Key Findings

### Strengths
1. ✅ Excellent thermodynamic property predictions
2. ✅ Strong metal classification (AUROC 0.9575)
3. ✅ Stable generalization (val ≈ test)

### Weaknesses
1. ⚠️ Moderate overfitting (train-val gap ~53%)
2. ⚠️ Electronic properties lag behind thermodynamic
3. ⚠️ Training continued 12 epochs past best validation

## Lessons Learned

1. Graph cache significantly improves data loading
2. Gradient clipping prevents NaN issues
3. Excluding volume/density/is_stable is critical
4. Need stronger regularization for Phase 2

## Files

- `model_checkpoint.pt` - Best model weights (epoch 38)
- `config.json` - Full training configuration
- `tensorboard/` - Training curves and metrics
- `analysis/` - Post-training analysis and visualizations

## Next Steps

See [Phase 2 Training Plan](../../reports/PHASE2_TRAINING_PLAN.md) for improvements.
