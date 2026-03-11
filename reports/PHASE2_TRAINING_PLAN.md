# Phase 2 Training Plan

**Date**: 2026-03-06
**Status**: 🚀 Ready to Start

---

## Overview

Phase 2 focuses on improving model performance by addressing Phase 1's identified weaknesses:
1. Reducing overfitting through regularization
2. Enhancing graph architecture with angle features
3. Improving training stability
4. Boosting electronic property predictions

---

## Experiment Organization Structure

### Directory Structure

```
experiments/
├── phase1/
│   ├── exp001_baseline_graph/
│   │   ├── config.json                    # Full training configuration
│   │   ├── model_checkpoint.pt            # Best model weights
│   │   ├── training_log.txt               # Console output
│   │   ├── tensorboard/                   # TensorBoard logs
│   │   ├── metrics/                       # Per-epoch metrics
│   │   ├── analysis/                      # Post-training analysis
│   │   │   ├── performance_report.md
│   │   │   ├── visualization.png
│   │   │   └── predictions.json
│   │   └── README.md                      # Experiment summary
│   └── summary.md                         # Phase 1 overall summary
│
├── phase2/
│   ├── exp002_regularization/             # Experiment 2: Add regularization
│   │   ├── config.json
│   │   ├── model_checkpoint.pt
│   │   ├── training_log.txt
│   │   ├── tensorboard/
│   │   ├── metrics/
│   │   ├── analysis/
│   │   └── README.md
│   │
│   ├── exp003_enhanced_graph/             # Experiment 3: Enhanced architecture
│   │   └── ...
│   │
│   ├── exp004_angle_features/             # Experiment 4: Add angle features
│   │   └── ...
│   │
│   └── summary.md                         # Phase 2 overall summary
│
└── comparison/
    ├── phase1_vs_phase2.md                # Cross-phase comparison
    └── best_models.md                     # Best model from each phase
```

### Naming Convention

**Experiment ID Format**: `expXXX_description`
- `XXX`: 3-digit sequential number (001, 002, 003, ...)
- `description`: Short descriptive name (lowercase, underscores)

**Examples**:
- `exp001_baseline_graph` - Phase 1 baseline
- `exp002_regularization` - Add dropout + weight decay
- `exp003_enhanced_graph` - Larger cutoff + more RBF
- `exp004_angle_features` - Add three-body interactions

---

## Phase 2 Experiment Plan

### Experiment 2: Enhanced Regularization
**Goal**: Reduce overfitting from Phase 1

**Changes from Baseline**:
- Weight decay: 1e-5 → **1e-4**
- Add dropout: **0.1** in prediction heads
- Early stopping: **patience=15**
- Learning rate: 0.0001 → **5e-5** (more conservative)

**Expected Outcome**:
- Reduce train-val gap by 30-50%
- Slight decrease in train performance
- Improved or maintained val/test performance

**Success Criteria**:
- Val loss < 0.22
- Train-val gap < 0.05

---

### Experiment 3: Enhanced Graph Architecture
**Goal**: Improve graph representation capacity

**Changes from Baseline**:
- Cutoff: 6.0 → **8.0 Å** (larger neighborhood)
- Max neighbors: 24 → **48**
- RBF basis: 64 → **128**
- Backbone: graph → **enhanced_graph**

**Expected Outcome**:
- Better capture of long-range interactions
- Improved electronic property predictions
- Potential increase in training time

**Success Criteria**:
- efermi MAE < 0.35 eV
- cbm/vbm MAE < 0.25 eV
- Overall val loss < 0.21

---

### Experiment 4: Angle Features (Three-Body Interactions)
**Goal**: Capture angular information for better geometry understanding

**Changes from Exp003**:
- Add `--use-angles` flag
- Enable three-body interaction terms

**Expected Outcome**:
- Better structural property predictions
- Improved band structure predictions (cbm/vbm)
- Longer training time per epoch

**Success Criteria**:
- band_gap MAE < 0.20 eV
- cbm/vbm MAE < 0.23 eV

---

### Experiment 5: Full Enhancement Stack
**Goal**: Combine best practices from Exp002-004

**Changes**:
- Enhanced graph architecture (Exp003)
- Angle features (Exp004)
- Regularization (Exp002)
- EMA (decay=0.999)
- Gradient clipping: 1.0 → **5.0**

**Expected Outcome**:
- Best overall performance
- Balanced train-val performance
- Production-ready model

**Success Criteria**:
- Val loss < 0.19
- All task MAE improvements > 10% vs baseline
- Train-val gap < 0.04

---

## Training Scripts

### Experiment 2: Regularization
```bash
python scripts/train_multitask.py \
  --split data/splits/split_iid_seed42.json \
  --stage a \
  --backbone graph \
  --batch-size 64 \
  --num-workers 8 \
  --exclude-tasks volume density is_stable \
  --epochs 100 \
  --lr 5e-5 \
  --weight-decay 1e-4 \
  --grad-clip 1.0 \
  --warmup-epochs 5 \
  --dropout 0.1 \
  --early-stopping-patience 15 \
  --experiment-name exp002_regularization
```

### Experiment 3: Enhanced Graph
```bash
python scripts/train_multitask.py \
  --split data/splits/split_iid_seed42.json \
  --stage a \
  --backbone enhanced_graph \
  --cutoff 8.0 \
  --max-neighbors 48 \
  --n-rbf 128 \
  --batch-size 64 \
  --num-workers 8 \
  --exclude-tasks volume density is_stable \
  --epochs 100 \
  --lr 2e-4 \
  --weight-decay 1e-5 \
  --grad-clip 5.0 \
  --warmup-epochs 5 \
  --experiment-name exp003_enhanced_graph
```

### Experiment 4: Angle Features
```bash
python scripts/train_multitask.py \
  --split data/splits/split_iid_seed42.json \
  --stage a \
  --backbone enhanced_graph \
  --cutoff 8.0 \
  --max-neighbors 48 \
  --n-rbf 128 \
  --use-angles \
  --batch-size 64 \
  --num-workers 8 \
  --exclude-tasks volume density is_stable \
  --epochs 100 \
  --lr 2e-4 \
  --weight-decay 1e-5 \
  --grad-clip 5.0 \
  --warmup-epochs 5 \
  --experiment-name exp004_angle_features
```

### Experiment 5: Full Stack
```bash
python scripts/train_multitask.py \
  --split data/splits/split_iid_seed42.json \
  --stage a \
  --backbone enhanced_graph \
  --cutoff 8.0 \
  --max-neighbors 48 \
  --n-rbf 128 \
  --use-angles \
  --use-edge-update \
  --batch-size 64 \
  --num-workers 8 \
  --exclude-tasks volume density is_stable \
  --epochs 100 \
  --lr 2e-4 \
  --weight-decay 1e-4 \
  --grad-clip 5.0 \
  --warmup-epochs 5 \
  --dropout 0.1 \
  --ema-decay 0.999 \
  --early-stopping-patience 15 \
  --experiment-name exp005_full_stack
```

---

## Evaluation Protocol

### After Each Experiment

1. **Training Monitoring**:
   - Monitor TensorBoard in real-time
   - Check for NaN/Inf issues
   - Verify GPU utilization

2. **Post-Training Analysis**:
   ```bash
   python scripts/analyze_best_model.py \
     --checkpoint experiments/stage_a/phase2_enhancements/expXXX_name/model_checkpoint.pt \
     --split data/splits/split_iid_seed42.json \
     --output-dir experiments/stage_a/phase2_enhancements/expXXX_name/analysis
   ```

3. **Document Results**:
   - Update experiment README.md
   - Record key metrics in phase2/summary.md
   - Note any unexpected behaviors

4. **Compare with Baseline**:
   - Generate comparison plots
   - Calculate improvement percentages
   - Identify which tasks improved/degraded

---

## Success Metrics

### Primary Metrics (Must Improve)
- Overall val loss < 0.20
- Train-val gap < 0.05
- efermi MAE < 0.35 eV

### Secondary Metrics (Nice to Have)
- band_gap MAE < 0.20 eV
- cbm/vbm MAE < 0.25 eV
- is_metal AUROC > 0.96

### Stability Metrics
- No NaN/Inf during training
- Gradient norm < 50 (99th percentile)
- Training completes without crashes

---

## Timeline

**Week 1** (Current):
- ✅ Complete Phase 1 analysis
- ✅ Design Phase 2 plan
- 🚀 Reorganize experiment structure
- 🚀 Run Experiment 2 (Regularization)

**Week 2**:
- Run Experiment 3 (Enhanced Graph)
- Run Experiment 4 (Angle Features)
- Compare Exp2-4 results

**Week 3**:
- Run Experiment 5 (Full Stack)
- Comprehensive analysis
- Select best model for production

---

## Risk Mitigation

### Potential Issues

1. **Longer Training Time**:
   - Enhanced graph + angles may be 2-3x slower
   - Mitigation: Use graph cache, optimize DataLoader

2. **Memory Issues**:
   - Larger cutoff increases memory usage
   - Mitigation: Reduce batch size if needed

3. **Overfitting Persists**:
   - Regularization may not be enough
   - Mitigation: Try data augmentation, ensemble methods

4. **No Improvement**:
   - Architecture changes may not help
   - Mitigation: Analyze failure cases, try different approaches

---

## Next Actions

1. **Reorganize existing Phase 1 results** into new structure
2. **Implement missing features** (dropout, early stopping, EMA)
3. **Run Experiment 2** as first Phase 2 trial
4. **Monitor and iterate** based on results

---

## References

- Phase 1 Summary: [PHASE1_FINAL_SUMMARY.md](PHASE1_FINAL_SUMMARY.md)
- Phase 2 Optimization Plan: [reports/plans/PHASE2_OPTIMIZATION_PLAN.md](reports/plans/PHASE2_OPTIMIZATION_PLAN.md)
- Enhanced Backbone Tests: [tests/test_enhanced_backbones.py](../../tests/test_enhanced_backbones.py)
