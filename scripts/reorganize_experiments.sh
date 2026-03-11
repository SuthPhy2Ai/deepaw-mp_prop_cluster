#!/bin/bash
# Reorganize Phase 1 results into new experiment structure

set -e

echo "🔄 Reorganizing Phase 1 results into new experiment structure..."

# Create new directory structure
BASE_DIR="/home/sutianhao/data/mp-data-pipeline"
EXP_DIR="${BASE_DIR}/experiments"

mkdir -p "${EXP_DIR}/phase1/exp001_baseline_graph"
mkdir -p "${EXP_DIR}/phase1/exp001_baseline_graph/analysis"
mkdir -p "${EXP_DIR}/phase2"
mkdir -p "${EXP_DIR}/comparison"

# Move Phase 1 artifacts
echo "📦 Moving Phase 1 training artifacts..."

# Copy checkpoint
if [ -f "${BASE_DIR}/artifacts/runs/20260305_210307/checkpoints/best.pt" ]; then
    cp "${BASE_DIR}/artifacts/runs/20260305_210307/checkpoints/best.pt" \
       "${EXP_DIR}/phase1/exp001_baseline_graph/model_checkpoint.pt"
    echo "  ✅ Checkpoint copied"
fi

# Copy config
if [ -f "${BASE_DIR}/artifacts/runs/20260305_210307/config.json" ]; then
    cp "${BASE_DIR}/artifacts/runs/20260305_210307/config.json" \
       "${EXP_DIR}/phase1/exp001_baseline_graph/config.json"
    echo "  ✅ Config copied"
fi

# Copy TensorBoard logs
if [ -d "${BASE_DIR}/artifacts/runs/20260305_210307/tensorboard" ]; then
    cp -r "${BASE_DIR}/artifacts/runs/20260305_210307/tensorboard" \
       "${EXP_DIR}/phase1/exp001_baseline_graph/"
    echo "  ✅ TensorBoard logs copied"
fi

# Copy metrics
if [ -d "${BASE_DIR}/artifacts/runs/20260305_210307/metrics" ]; then
    cp -r "${BASE_DIR}/artifacts/runs/20260305_210307/metrics" \
       "${EXP_DIR}/phase1/exp001_baseline_graph/"
    echo "  ✅ Metrics copied"
fi

# Move analysis results
echo "📊 Moving analysis results..."

if [ -d "${BASE_DIR}/reports/model_analysis_20260306" ]; then
    cp "${BASE_DIR}/reports/model_analysis_20260306/analysis_report.md" \
       "${EXP_DIR}/phase1/exp001_baseline_graph/analysis/performance_report.md"
    cp "${BASE_DIR}/reports/model_analysis_20260306/performance_visualization.png" \
       "${EXP_DIR}/phase1/exp001_baseline_graph/analysis/visualization.png"
    cp "${BASE_DIR}/reports/model_analysis_20260306/results.json" \
       "${EXP_DIR}/phase1/exp001_baseline_graph/analysis/predictions.json"
    echo "  ✅ Analysis results copied"
fi

# Create experiment README
echo "📝 Creating experiment README..."

cat > "${EXP_DIR}/phase1/exp001_baseline_graph/README.md" << 'EOF'
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
EOF

echo "  ✅ README created"

# Create Phase 1 summary
echo "📋 Creating Phase 1 summary..."

cp "${BASE_DIR}/reports/PHASE1_FINAL_SUMMARY.md" \
   "${EXP_DIR}/phase1/summary.md"

echo "  ✅ Phase 1 summary copied"

# Create symlinks to keep old paths working
echo "🔗 Creating symlinks for backward compatibility..."

if [ ! -L "${BASE_DIR}/artifacts/runs/20260305_210307/exp001_link" ]; then
    ln -s "${EXP_DIR}/phase1/exp001_baseline_graph" \
       "${BASE_DIR}/artifacts/runs/20260305_210307/exp001_link"
    echo "  ✅ Symlink created"
fi

# Create experiment tracking file
echo "📊 Creating experiment tracking file..."

cat > "${EXP_DIR}/EXPERIMENTS.md" << 'EOF'
# Experiment Tracking

This file tracks all experiments across phases.

## Phase 1: Baseline

| ID | Name | Status | Val Loss | Best Epoch | Notes |
|----|------|--------|----------|------------|-------|
| exp001 | baseline_graph | ✅ Complete | 0.2226 | 38 | Baseline SchNet-style model |

## Phase 2: Enhancements

| ID | Name | Status | Val Loss | Best Epoch | Notes |
|----|------|--------|----------|------------|-------|
| exp002 | regularization | 🚀 Planned | - | - | Add dropout + weight decay |
| exp003 | enhanced_graph | 🚀 Planned | - | - | Larger cutoff + more RBF |
| exp004 | angle_features | 🚀 Planned | - | - | Three-body interactions |
| exp005 | full_stack | 🚀 Planned | - | - | Combined enhancements |

## Legend

- ✅ Complete
- 🚀 Planned
- 🔄 Running
- ⚠️ Failed
- 🔍 Analyzing

## Best Models

| Phase | Experiment | Val Loss | Test Loss | Notes |
|-------|------------|----------|-----------|-------|
| 1 | exp001_baseline_graph | 0.2226 | 0.2223 | Baseline |

EOF

echo "  ✅ Experiment tracking file created"

echo ""
echo "✅ Reorganization complete!"
echo ""
echo "📁 New structure:"
echo "   ${EXP_DIR}/phase1/exp001_baseline_graph/"
echo "   ${EXP_DIR}/phase2/ (ready for new experiments)"
echo "   ${EXP_DIR}/comparison/ (for cross-phase analysis)"
echo ""
echo "📖 Documentation:"
echo "   ${EXP_DIR}/EXPERIMENTS.md - Experiment tracking"
echo "   ${EXP_DIR}/phase1/summary.md - Phase 1 summary"
echo "   ${BASE_DIR}/reports/PHASE2_TRAINING_PLAN.md - Phase 2 plan"
echo ""
echo "🚀 Ready to start Phase 2!"
