# Improved Experiment Directory Structure

## Problem
Current structure doesn't distinguish between Stage A (8 tasks) and Stage B (18 tasks including elastic properties).

## Proposed Structure

```
experiments/
├── stage_a/                           # Stage A: 8 core tasks (no elastic)
│   ├── phase1_baseline/
│   │   └── exp001_baseline_graph/
│   ├── phase2_enhancements/
│   │   ├── exp002_regularization/
│   │   ├── exp003_enhanced_graph/
│   │   ├── exp004_angle_features/
│   │   └── exp005_full_stack/
│   └── summary.md
│
├── stage_b/                           # Stage B: 18 tasks (with elastic)
│   ├── phase3_baseline/
│   │   └── exp101_baseline_graph/     # Re-train baseline with Stage B
│   ├── phase3_enhancements/
│   │   ├── exp102_regularization/
│   │   ├── exp103_enhanced_graph/
│   │   └── exp104_full_stack/
│   └── summary.md
│
├── comparison/
│   ├── stage_a_best_models.md
│   ├── stage_b_best_models.md
│   └── stage_a_vs_b.md
│
├── EXPERIMENTS.md                     # Master tracking table
└── README.md
```

## Naming Convention

### Stage A Experiments (8 tasks)
- **exp001-099**: Stage A experiments
- Phase 1: exp001-009 (baseline)
- Phase 2: exp010-099 (enhancements)

### Stage B Experiments (18 tasks)
- **exp101-199**: Stage B experiments
- Phase 3: exp101-109 (baseline)
- Phase 4: exp110-199 (enhancements)

## Benefits

1. **Clear Stage Separation**: Stage A and Stage B experiments are in separate directories
2. **Scalable**: Easy to add more phases within each stage
3. **Traceable**: Experiment ID ranges indicate stage (001-099 = Stage A, 101-199 = Stage B)
4. **Comparable**: Easy to compare same model architecture across stages
5. **Future-proof**: Can add Stage C (e.g., with additional tasks) as `stage_c/`

## Migration Plan

### Step 1: Reorganize Current Experiments
Move Phase 1 and Phase 2 experiments to `stage_a/` structure:
- `phase1/exp001_baseline_graph/` → `stage_a/phase1_baseline/exp001_baseline_graph/`
- `phase2/exp002_*` → `stage_a/phase2_enhancements/exp002_*/`

### Step 2: Update Configuration Files
Update experiment configs to reflect new paths.

### Step 3: Update Tools
Modify `experiment_manager.py` to support stage-aware operations.

### Step 4: Update Documentation
Update all references to new structure in:
- docs/guides/PHASE2_QUICKSTART.md
- PHASE2_TRAINING_PLAN.md
- experiments/README.md

## Alternative: Simpler Structure

If the above is too complex, a simpler alternative:

```
experiments/
├── exp001_baseline_graph_stageA/      # Explicit stage in name
├── exp002_regularization_stageA/
├── exp003_enhanced_graph_stageA/
├── exp101_baseline_graph_stageB/      # Stage B starts at 101
├── exp102_regularization_stageB/
└── ...
```

This keeps flat structure but uses naming convention to distinguish stages.

## Recommendation

I recommend the **hierarchical structure** (first option) because:
1. Clearer visual separation
2. Easier to navigate
3. Better for comparison within same stage
4. More maintainable as project grows

What do you think? Should I proceed with the migration?
