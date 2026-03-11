# Experiments Directory

This directory contains all training experiments organized by stage and phase.

## Structure

```
experiments/
├── stage_a/                           # Stage A: 8 core tasks (no elastic)
│   ├── phase1_baseline/
│   │   └── exp001_baseline_graph/     # Baseline graph model
│   ├── phase2_enhancements/
│   │   ├── exp002_regularization/     # Enhanced regularization
│   │   ├── exp003_enhanced_graph/     # Enhanced architecture
│   │   ├── exp004_angle_features/     # Angle features
│   │   └── exp005_full_stack/         # Combined enhancements
│   └── summary.md                     # Stage A summary
│
├── stage_b/                           # Stage B: 18 tasks (with elastic)
│   ├── phase3_baseline/
│   │   └── exp101_baseline_graph/     # Re-train baseline with Stage B
│   ├── phase3_enhancements/
│   │   ├── exp102_regularization/
│   │   ├── exp103_enhanced_graph/
│   │   └── exp104_full_stack/
│   └── summary.md                     # Stage B summary
│
├── comparison/                        # Cross-stage comparisons
│   ├── stage_a_best_models.md
│   ├── stage_b_best_models.md
│   └── stage_a_vs_b.md
│
├── EXPERIMENTS.md                     # Master tracking table
└── README.md                          # This file
```

## Experiment Naming Convention

### Stage A Experiments (8 tasks)
**Format**: `expXXX_description` where XXX = 001-099

- **Phase 1** (exp001-009): Baseline models
  - exp001: Baseline graph model
- **Phase 2** (exp010-099): Enhanced models
  - exp002-005: Regularization, architecture, features

### Stage B Experiments (18 tasks including elastic)
**Format**: `expXXX_description` where XXX = 101-199

- **Phase 3** (exp101-109): Baseline models with Stage B
  - exp101: Baseline graph model (Stage B)
- **Phase 4** (exp110-199): Enhanced models with Stage B
  - exp102-105: Same enhancements as Phase 2, but with elastic tasks

## Standard Experiment Structure

Each experiment directory contains:

```
expXXX_name/
├── README.md                 # Experiment documentation
├── config.json              # Training configuration
├── model_checkpoint.pt      # Best model weights
├── training_log.txt         # Console output
├── tensorboard/             # TensorBoard logs
├── metrics/                 # Per-epoch metrics
└── analysis/                # Post-training analysis
    ├── performance_report.md
    ├── visualization.png
    └── predictions.json
```

## Quick Start

### List All Experiments

```bash
python scripts/experiment_manager.py list
```

### Create New Experiment

```bash
# Stage A experiment
python scripts/experiment_manager.py create exp006 my_experiment --stage a

# Stage B experiment
python scripts/experiment_manager.py create exp101 my_experiment --stage b
```

### Run Analysis

```bash
python scripts/experiment_manager.py analyze exp002 regularization
```

## Stage Comparison

### Stage A (8 tasks)
- energy_per_atom
- formation_energy_per_atom
- energy_above_hull
- band_gap
- cbm
- vbm
- efermi
- is_metal

**Coverage**: ~155k materials

### Stage B (18 tasks)
All Stage A tasks plus:
- volume
- density
- is_stable
- bulk_modulus_vrh
- shear_modulus_vrh
- youngs_modulus
- homogeneous_poisson
- universal_anisotropy

**Coverage**: ~155k materials (but only ~11k with elastic data)

## Documentation

- **Stage A Summary**: `stage_a/summary.md`
- **Stage B Summary**: `stage_b/summary.md`
- **Phase 2 Plan**: `../reports/PHASE2_TRAINING_PLAN.md`
- **Quick Start Guide**: `../PHASE2_QUICKSTART.md`
- **Experiment Tracking**: `EXPERIMENTS.md`

## Best Practices

1. **Always document**: Update README.md after each experiment
2. **Track progress**: Update EXPERIMENTS.md with results
3. **Save checkpoints**: Copy best.pt to experiment directory
4. **Run analysis**: Generate comprehensive analysis reports
5. **Compare results**: Document improvements vs baseline
6. **Stage awareness**: Clearly indicate which stage (A or B) in all documentation

## Current Status

- ✅ Stage A Phase 1: Complete (1 experiment)
- 🚀 Stage A Phase 2: Ready to start (4 experiments planned)
- 📋 Stage B Phase 3: Planned for future

See `EXPERIMENTS.md` for detailed tracking.
