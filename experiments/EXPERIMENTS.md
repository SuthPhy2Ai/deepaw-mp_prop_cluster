# Experiment Tracking

This file tracks all experiments across stages and phases.

## Stage A: 8 Core Tasks (No Elastic Properties)

### Phase 1: Baseline

| ID | Name | Status | Val Loss | Best Epoch | Notes |
|----|------|--------|----------|------------|-------|
| exp001 | baseline_graph | ✅ Complete | 0.2226 | 38 | Baseline SchNet-style model |

### Phase 2: Enhancements

| ID | Name | Status | Val Loss | Best Epoch | Notes |
|----|------|--------|----------|------------|-------|
| exp002 | regularization | 🚀 Planned | - | - | Add dropout + weight decay |
| exp003 | enhanced_graph | 🚀 Planned | - | - | Larger cutoff + more RBF |
| exp004 | angle_features | 🚀 Planned | - | - | Three-body interactions |
| exp005 | full_stack | 🚀 Planned | - | - | Combined enhancements |

## Stage B: 18 Tasks (With Elastic Properties)

### Phase 3: Baseline

| ID | Name | Status | Val Loss | Best Epoch | Notes |
|----|------|--------|----------|------------|-------|
| exp101 | baseline_graph | 📋 Future | - | - | Baseline with elastic tasks |

### Phase 3: Enhancements

| ID | Name | Status | Val Loss | Best Epoch | Notes |
|----|------|--------|----------|------------|-------|
| exp102 | regularization | 📋 Future | - | - | Regularization with elastic |
| exp103 | enhanced_graph | 📋 Future | - | - | Enhanced graph with elastic |
| exp104 | full_stack | 📋 Future | - | - | Full stack with elastic |

## Legend

- ✅ Complete
- 🚀 Planned (ready to start)
- 📋 Future (not yet planned in detail)
- 🔄 Running
- ⚠️ Failed
- 🔍 Analyzing

## Best Models by Stage

### Stage A (8 tasks)

| Phase | Experiment | Val Loss | Test Loss | Notes |
|-------|------------|----------|-----------|-------|
| 1 | exp001_baseline_graph | 0.2226 | 0.2223 | Baseline |

### Stage B (18 tasks)

| Phase | Experiment | Val Loss | Test Loss | Notes |
|-------|------------|----------|-----------|-------|
| - | - | - | - | Not started yet |

## Task Coverage by Stage

### Stage A Tasks (8)
- energy_per_atom
- formation_energy_per_atom
- energy_above_hull
- band_gap
- cbm
- vbm
- efermi
- is_metal

### Stage B Additional Tasks (10)
- volume
- density
- is_stable
- bulk_modulus_vrh
- shear_modulus_vrh
- youngs_modulus
- homogeneous_poisson
- universal_anisotropy
- (2 more elastic properties)

## Notes

- Stage A experiments use `--exclude-tasks volume density is_stable` plus elastic tasks
- Stage B experiments include all 18 tasks with `--oversample-elastic 4.0`
- Experiment IDs 001-099 reserved for Stage A
- Experiment IDs 101-199 reserved for Stage B

