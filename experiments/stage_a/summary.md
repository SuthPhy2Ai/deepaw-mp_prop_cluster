# Stage A Summary

**Stage**: A (8 core tasks, no elastic properties)
**Status**: Phase 1 complete, Phase 2 ready to start

---

## Tasks Included (8)

### Thermodynamic (3)
- energy_per_atom
- formation_energy_per_atom
- energy_above_hull

### Electronic (5)
- band_gap (regression)
- cbm (conduction band minimum)
- vbm (valence band maximum)
- efermi (Fermi energy)
- is_metal (classification)

---

## Phase 1: Baseline (Complete)

### exp001_baseline_graph

**Status**: ✅ Complete
**Best Epoch**: 38
**Val Loss**: 0.2226
**Test Loss**: 0.2223

#### Key Metrics (Validation Set)

**Best Tasks**:
- formation_energy_per_atom: MAE = 0.0800 eV, R² = 0.988 ⭐
- energy_above_hull: MAE = 0.0644 eV, R² = 0.924 ⭐
- is_metal: AUROC = 0.9575 ⭐

**Needs Improvement**:
- efermi: MAE = 0.3834 eV ⚠️
- energy_per_atom: MAE = 0.3606 eV ⚠️
- cbm: MAE = 0.2921 eV ⚠️

**Overfitting Analysis**:
- Overall train-val gap: 53% ⚠️
- energy_per_atom gap: 95% ⚠️
- cbm gap: 57% ⚠️
- band_gap gap: 61% ⚠️

#### Lessons Learned
1. Strong performance on thermodynamic properties
2. Significant overfitting on electronic properties
3. Need better regularization
4. Graph architecture could be enhanced

---

## Phase 2: Enhancements (Planned)

### exp002_regularization
**Goal**: Reduce overfitting
**Changes**: dropout=0.1, weight_decay=1e-4, lr=5e-5, early_stopping
**Expected**: Train-val gap < 0.05, Val loss < 0.22

### exp003_enhanced_graph
**Goal**: Better graph representation
**Changes**: cutoff=8.0Å, max_neighbors=48, n_rbf=128
**Expected**: efermi MAE < 0.35 eV, Val loss < 0.21

### exp004_angle_features
**Goal**: Capture three-body interactions
**Changes**: Based on exp003 + angle features
**Expected**: band_gap MAE < 0.20 eV, cbm/vbm MAE < 0.23 eV

### exp005_full_stack
**Goal**: Production-ready model
**Changes**: All enhancements combined
**Expected**: Val loss < 0.19, all tasks improve >10%

---

## Next Steps

1. Launch exp002_regularization
2. Monitor training and compare with baseline
3. Iterate through exp003-005
4. Select best model for Stage A
5. Consider moving to Stage B (with elastic properties)

---

## Files

- Baseline results: `phase1_baseline/exp001_baseline_graph/`
- Phase 2 configs: `/scratch/sutianhao/data/mp-data-pipeline/configs/exp002-005_*.json`
- Detailed plan: `/scratch/sutianhao/data/mp-data-pipeline/reports/PHASE2_TRAINING_PLAN.md`
