# Experiment 208: DeePAW Zero Single-Task Family

**Date**: 2026-03-14
**Status**: Ready
**Phase**: Phase 2 DeePAW Integration
**Experiment ID**: exp208_deepaw_zero_single_task

## Objective

Train 15 fully independent models (one per property) using **Enhanced Graph + DeePAW Replace mode** with no shared backbone. This experiment benchmarks DeePAW's effectiveness against the pure Graph baseline (exp106).

**Research Question**: Does DeePAW improve single-task performance when models are trained independently (no multi-task learning)?

## Comparison with exp106

| Aspect | exp106 (Baseline) | exp208 (DeePAW) |
|--------|-------------------|-----------------|
| Backbone | `graph` (SchNet-style) | `enhanced_graph` |
| Atom Embeddings | Learnable embeddings | DeePAW pretrained features (Replace mode) |
| RBF Basis | 64 | 128 |
| Capacity Policy | Same (160-320 dim) | Same (160-320 dim) |
| Training Strategy | Single-task, independent | Single-task, independent |

**Key Difference**: Only the atom embedding source differs - exp106 uses learnable embeddings, exp208 uses DeePAW pretrained charge density features.

## Scope

**15 Tasks** (same as exp106):
- **8 Stage A core tasks**: energy_per_atom, formation_energy_per_atom, energy_above_hull, band_gap, cbm, vbm, efermi, is_metal
- **3 excluded from Stage A multi-task**: volume, density, is_stable
- **4 elastic tasks**: bulk_modulus_vrh, shear_modulus_vrh, homogeneous_poisson, universal_anisotropy

## Architecture

### Backbone: Enhanced Graph + DeePAW Replace

```
Input (Crystal Structure)
    ↓
Enhanced Graph Backbone:
  - Cutoff: 6.0Å
  - Max neighbors: 24
  - RBF basis: 128 (vs 64 in exp106)
  - DeePAW features: Replace mode
    → DeePAW (3200-dim) → Projection → hidden_dim
    → Skips learnable atom embeddings entirely
    ↓
Message Passing (GNN layers)
    ↓
Task-Specific Head (single task)
    ↓
Output (1 property prediction)
```

### DeePAW Configuration

- **Checkpoint**: `/home/sutianhao/data/deepaw_test/DeePAW-main/checkpoints/f_nonlocal_escn_best.pth`
- **Fusion Mode**: `replace` (most aggressive - DeePAW replaces atom embeddings)
- **Frozen**: DeePAW model weights are frozen, only projection layer is trainable
- **Features**: 3200-dim pretrained charge density embeddings from 1.8M DFT calculations

## Capacity Policy (Identical to exp106)

Model capacity is sized per task according to sample count and task characteristics:

| Tier | Condition | Hidden Dim | Layers | Batch Size | Epochs | Learning Rate | Tasks |
|------|-----------|-----------|--------|-----------|--------|---------------|-------|
| **Special** | volume, density | 320 | 7 | 32 | 70 | 8e-5 | 2 tasks |
| **Tier 1** | ≥100k samples | 256 | 6 | 64 | 50 | 1e-4 | 6 tasks |
| **Tier 2** | ≥50k samples | 224 | 6 | 64 | 55 | 1e-4 | 2 tasks |
| **Tier 3** | <50k samples | 160 | 4 | 32 | 80 | 2e-4 | 4 tasks |
| **Special** | homogeneous_poisson, universal_anisotropy | 160 | 4 | 32 | 80 | 2e-4 | (use --no-amp) |

### Task Mapping

**Special (320×7)**:
- volume (123,890 samples)
- density (123,902 samples)

**Tier 1 (256×6)**:
- energy_per_atom (123,902 samples)
- formation_energy_per_atom (123,902 samples)
- energy_above_hull (123,902 samples)
- band_gap (123,902 samples)
- efermi (123,856 samples)
- is_metal (123,902 samples)
- is_stable (123,902 samples)

**Tier 2 (224×6)**:
- cbm (71,574 samples)
- vbm (71,574 samples)

**Tier 3 (160×4)**:
- bulk_modulus_vrh (10,217 samples)
- shear_modulus_vrh (9,685 samples)
- homogeneous_poisson (9,752 samples) - with --no-amp
- universal_anisotropy (9,057 samples) - with --no-amp

## Output Isolation

- **Experiment root**: `experiments/stage_a/phase2_deepaw/exp208_deepaw_zero_single_task/`
- **Run root**: `artifacts/runs_exp208/<task>/<timestamp>/`
- **Logs**: `experiments/stage_a/phase2_deepaw/exp208_deepaw_zero_single_task/logs/<task>.log`
- **Metrics**: `experiments/stage_a/phase2_deepaw/exp208_deepaw_zero_single_task/metrics/exp208_results.csv`

Each task has completely isolated:
- Training command script: `tasks/<task>/training_cmd.sh`
- Model checkpoints: `artifacts/runs_exp208/<task>/<timestamp>/checkpoints/`
- Training logs: `logs/<task>.log`
- Metrics: Aggregated in `metrics/exp208_results.csv`

## Training Configuration

### Common Parameters (All Tasks)

```bash
--backbone enhanced_graph
--use-deepaw-features
--deepaw-checkpoint /home/sutianhao/data/deepaw_test/DeePAW-main/checkpoints/f_nonlocal_escn_best.pth
--deepaw-fusion replace
--cutoff 6.0
--max-neighbors 24
--n-rbf 128
--weight-decay 1e-05
--num-workers 4
--warmup-epochs 5
--device cuda
--use-pyg
```

### Task-Specific Parameters

Vary by capacity tier:
- `--hidden-dim`: 160, 224, 256, or 320
- `--layers`: 4, 6, or 7
- `--batch-size`: 32 or 64
- `--epochs`: 50, 55, 70, or 80
- `--lr`: 8e-5, 1e-4, or 2e-4
- `--grad-clip`: 0.5, 0.8, or 1.0
- `--no-amp`: Only for homogeneous_poisson, universal_anisotropy

## Hypothesis

**DeePAW pretrained features will improve single-task performance, especially for electronic properties.**

### Expected Improvements over exp106

| Property Type | Expected Improvement | Rationale |
|---------------|---------------------|-----------|
| **Electronic** (band_gap, cbm, vbm, efermi) | 10-20% MAE reduction | DeePAW's charge density pretraining directly encodes electronic structure |
| **Thermodynamic** (energy_per_atom, formation_energy_per_atom, energy_above_hull) | 5-10% MAE reduction | Charge density correlates with bonding and stability |
| **Structural** (volume, density) | Minimal improvement | Geometric properties, less dependent on electronic structure |
| **Elastic** (bulk_modulus_vrh, shear_modulus_vrh, etc.) | Minimal improvement | Mechanical properties, low data coverage limits gains |
| **Classification** (is_metal, is_stable) | 2-5% AUROC improvement | Electronic structure helps distinguish metallic/stable materials |

### Success Criteria

1. **All 15 tasks train successfully** without errors
2. **At least 10/15 tasks show improvement** over exp106
3. **Electronic properties show strongest gains** (validates DeePAW's charge density pretraining)
4. **No degradation on structural/elastic tasks** (DeePAW should not hurt performance)

## Execution

### Orchestration Script

**File**: `scripts/run_exp208_deepaw_zero.py`

**Usage**:
```bash
# Dry-run (generate scripts only, no training)
python scripts/run_exp208_deepaw_zero.py --dry-run

# Train all 15 tasks sequentially
python scripts/run_exp208_deepaw_zero.py

# Train specific tasks only
python scripts/run_exp208_deepaw_zero.py --tasks band_gap cbm vbm

# Train first N tasks (for testing)
python scripts/run_exp208_deepaw_zero.py --max-tasks 3
```

### Manual Execution

Each task can be trained independently:
```bash
cd /scratch/sutianhao/data/mp-data-pipeline
bash experiments/stage_a/phase2_deepaw/exp208_deepaw_zero_single_task/tasks/band_gap/training_cmd.sh
```

### Parallel Execution

If multiple GPUs available, tasks can be parallelized:
```bash
# GPU 0: Tier 1 tasks (high capacity)
CUDA_VISIBLE_DEVICES=0 bash tasks/band_gap/training_cmd.sh &
CUDA_VISIBLE_DEVICES=0 bash tasks/energy_per_atom/training_cmd.sh &

# GPU 1: Tier 2/3 tasks (lower capacity)
CUDA_VISIBLE_DEVICES=1 bash tasks/cbm/training_cmd.sh &
CUDA_VISIBLE_DEVICES=1 bash tasks/bulk_modulus_vrh/training_cmd.sh &
```

## Timeline Estimate

**Sequential execution** (single GPU):
- Tier 1 tasks (7 tasks × 50 epochs × ~40 min): ~4.7 hours
- Tier 2 tasks (2 tasks × 55 epochs × ~45 min): ~1.5 hours
- Tier 3 tasks (4 tasks × 80 epochs × ~30 min): ~2 hours
- Special tasks (2 tasks × 70 epochs × ~50 min): ~1.7 hours
- **Total**: ~10 hours

**Parallel execution** (2 GPUs):
- **Total**: ~5-6 hours

## Comparison Metrics

After training, compare with exp106 using:

### Regression Tasks
- **MAE** (Mean Absolute Error) - primary metric
- **R²** (Coefficient of Determination)
- **RMSE** (Root Mean Squared Error)

### Classification Tasks
- **AUROC** (Area Under ROC Curve) - primary metric
- **Accuracy**
- **F1 Score**

### Analysis
- **Per-task improvement**: (exp106_MAE - exp208_MAE) / exp106_MAE × 100%
- **Property type analysis**: Average improvement by category (electronic, thermodynamic, etc.)
- **Capacity correlation**: Does improvement vary with model size?
- **Data coverage correlation**: Does improvement vary with training sample count?

## Key Design Decisions

### 1. Why Enhanced Graph instead of Graph?

- DeePAW integration is only implemented in `EnhancedGraphBackbone`
- Enhanced Graph has better architecture (128 RBF basis vs 64)
- Maintains consistency with exp201-207 DeePAW experiments
- **Trade-off**: Not a perfect apples-to-apples comparison with exp106, but necessary for DeePAW integration

### 2. Why Replace mode?

- **Most aggressive** DeePAW integration (completely replaces atom embeddings)
- Tests if DeePAW alone is sufficient without learnable embeddings
- Provides **clearest comparison**: Graph embeddings vs DeePAW embeddings
- **Alternative modes** (add, concat) could be explored in future experiments

### 3. Why same capacity policy as exp106?

- **Fair comparison** requires identical model capacity
- Only difference should be embedding source (Graph vs DeePAW)
- Allows **direct performance attribution** to DeePAW features
- Eliminates confounding factors (model size, training epochs, etc.)

### 4. Why 15 tasks instead of 8 Stage A tasks?

- **Comprehensive benchmark** across all property types
- Tests DeePAW on low-coverage tasks (elastic properties)
- Matches exp106 scope for **complete comparison**
- Enables analysis of DeePAW's effectiveness across different data regimes

## Expected Outcomes

### Best Case Scenario

- Electronic properties: 15-20% MAE reduction (band_gap < 0.60 eV vs exp106's ~0.70 eV)
- Thermodynamic properties: 8-12% MAE reduction
- Classification tasks: 3-5% AUROC improvement
- **Conclusion**: DeePAW pretrained features significantly improve single-task performance

### Worst Case Scenario

- No improvement or slight degradation across all tasks
- DeePAW features don't transfer well to single-task setting
- **Conclusion**: DeePAW benefits require multi-task learning to emerge

### Most Likely Scenario

- Electronic properties: 10-15% improvement (validates charge density pretraining)
- Thermodynamic properties: 5-8% improvement
- Structural/elastic properties: Minimal change (±2%)
- **Conclusion**: DeePAW provides targeted improvements for electronic structure prediction

## Next Steps

After exp208 completes:

1. **Compare with exp106**: Generate side-by-side performance tables
2. **Analyze improvement patterns**: Which property types benefit most?
3. **Investigate failures**: Which tasks show no improvement or degradation?
4. **Explore alternative fusion modes**: Test Add/Concat modes if Replace underperforms
5. **Multi-task comparison**: Compare exp208 (single-task + DeePAW) with exp201-207 (multi-task + DeePAW)

## References

- **Baseline**: [exp106_zero_single_task_family](../../zero_version/exp106_zero_single_task_family/)
- **DeePAW Integration**: [docs/deepaw_integration/](../../../docs/deepaw_integration/)
- **Multi-task DeePAW**: [exp201-207](../) (Phase 2 DeePAW experiments)
- **Training Script**: [scripts/run_exp208_deepaw_zero.py](../../../scripts/run_exp208_deepaw_zero.py)
