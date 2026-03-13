# Experiments Directory

This directory contains all multitask learning experiments for materials property prediction. Each experiment trains a single model to predict multiple properties simultaneously using shared feature extraction.

## Multitask Learning Architecture

All experiments follow this architecture:
```
Input (Crystal Structure)
    ↓
Backbone (Shared Feature Extractor)
    ├─ Graph Neural Network (message passing on atomic graph)
    ├─ OR Composition Embedding (element features only)
    ├─ OR Enhanced Graph (GNN + angle features + DeePAW)
    ↓
Shared Representation (learned features)
    ↓
Task-Specific Heads (separate output layers)
    ├─ Thermodynamic Head → energy_per_atom, formation_energy_per_atom, energy_above_hull
    ├─ Electronic Head → band_gap, cbm, vbm, efermi, is_metal
    ├─ Structural Head → volume, density
    ├─ Stability Head → is_stable
    └─ Elastic Head → bulk_modulus_vrh, shear_modulus_vrh, etc.
```

**Key Concept**: One model, multiple tasks. The backbone learns shared representations that benefit all tasks through joint training.

## Directory Structure

```
experiments/
├── stage_a/                           # Stage A: 8 core tasks (no elastic)
│   ├── phase1_baseline/               # exp001-002: Initial baselines
│   │   ├── exp001_composition/        # Composition baseline
│   │   └── exp002_graph/              # Graph baseline (directory: exp001_baseline_graph)
│   ├── phase2_deepaw/                 # exp201-299: DeePAW integration
│   │   ├── exp201_deepaw_add/         # Add fusion
│   │   ├── exp202_deepaw_concat/      # Concat fusion
│   │   ├── exp203_deepaw_angles/      # With angle features
│   │   ├── exp204_deepaw_long/        # 100 epochs (exp204a)
│   │   ├── exp204_deepaw_replace/     # Replace fusion (exp204b)
│   │   ├── exp205_deepaw_lr1e4/       # Lower learning rate
│   │   └── exp207_deepaw_finetune/    # Two-stage fine-tuning
│   ├── phase2_enhancements/           # Reserved for future enhancements
│   └── summary.md                     # Stage A summary
│
├── stage_b/                           # Stage B: 18 tasks (with elastic)
│   ├── phase3_baseline/               # exp101: Stage B baseline
│   │   └── exp101_baseline_graph/     # Graph baseline with all 18 tasks
│   ├── phase2_deepaw/                 # exp206: DeePAW for Stage B
│   │   └── exp206_deepaw_stageb/      # DeePAW with elastic tasks
│   ├── phase3_enhancements/           # Reserved for Stage B enhancements
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

**IMPORTANT**: This project uses a hybrid numbering system. Read carefully to understand the ID ranges.

### Experiment ID Ranges

| ID Range | Stage | Phase | Description |
|----------|-------|-------|-------------|
| 001-099  | A | Phase 1 | Early baseline experiments (composition, graph) |
| 101-109  | B | Phase 3 | Stage B baseline and enhancements (all 18 tasks) |
| 105      | A | Phase 3 | PyG backend baseline for Stage A |
| 106      | Zero | - | Single-task family (no multitask, per-task capacity) |
| 107-109  | C | Phase 1 | Head architecture variants (hierarchical, derived) |
| 201-299  | A | Phase 2 | DeePAW feature integration experiments |
| 204a/204b| A | Phase 2 | **Special case**: Two exp204 variants (long training vs replace fusion) |
| 301-399  | - | - | Reserved for future experiments |

### Naming Conventions and Special Cases

**Directory vs Experiment ID Mismatches**:
- **exp002**: Directory named `exp001_baseline_graph` but is actually exp002 (Graph baseline)
  - exp001 = Composition baseline (run: 20260303_211013)
  - exp002 = Graph baseline (run: 20260304_005923, directory: exp001_baseline_graph)

**Experiment ID Conflicts**:
- **exp204**: Has two variants due to parallel development
  - **exp204a** (`exp204_deepaw_long`): 100 epochs training, Add fusion
  - **exp204b** (`exp204_deepaw_replace`): 50 epochs, Replace fusion mode
  - Both use the same output directory: `artifacts/runs_exp204`
  - Future experiments should avoid this pattern and use sequential IDs

### Zero Version (exp106)

**Concept**: Baseline without multitask learning - train completely independent models per task.

- **exp106 (Zero Single-Task Family)**:
  ```
  Per-Task Architecture:
  Input → Graph Backbone (独立) → Single Task Head

  No shared backbone, no multitask learning
  ```
  - **Capacity policy**: Model size based on data availability
    - volume, density: 320-dim, 7 layers (large values, need capacity)
    - ≥100k samples: 256-dim, 6 layers
    - ≥50k samples: 224-dim, 6 layers
    - <50k samples: 160-dim, 4 layers
  - **Purpose**: Establish upper bound - can multitask match single-task performance?
  - **Training**: 15 independent models, one per task
  - **Output**: `artifacts/runs_zero/<task>/<run_id>/`

### Stage A Experiments (8 core tasks)

**What it trains**: A single model that simultaneously predicts 8 materials properties:
1. **Thermodynamic** (3 tasks): energy_per_atom, formation_energy_per_atom, energy_above_hull
2. **Electronic** (5 tasks): band_gap, cbm, vbm, efermi (regression), is_metal (classification)

**Why exclude volume/density/is_stable**: These tasks cause training instability (volume values range 5-10,000, causing gradient explosion and NaN loss).

**Architecture**: Shared backbone → Task-specific heads (thermodynamic head + electronic head)

**Experiment Series**:

#### Phase 1 Baseline (exp001-002)
- **exp001 (Composition)**: Element embedding backbone only, no structure
  - Backbone: 118-element embedding → MLP
  - Run ID: 20260303_211013
  - Result: Won 9/11 tasks, strong baseline
- **exp002 (Graph)**: SchNet-style message passing
  - Backbone: GNN (6.0Å cutoff, 24 neighbors, 64 RBF basis)
  - Run ID: 20260304_005923
  - Directory: `exp001_baseline_graph` (naming mismatch, actually exp002)
  - Result: Won 2/11 tasks, underperformed expectations

#### Phase 2 DeePAW Integration (exp201-299)
All experiments use **enhanced_graph backbone** with DeePAW pretrained atomic features:
- **Base architecture**: GNN (8.0Å cutoff, 48 neighbors, 128 RBF basis)
- **DeePAW features**: Pretrained charge density embeddings from 1.8M DFT calculations
- **Fusion strategies**: How to combine composition + structure + DeePAW features

**Experiments**:
- **exp201 (Add Fusion)**:
  ```
  Composition Embedding (128-dim)
       +
  Structure Embedding (128-dim from GNN)
       +
  DeePAW Features (128-dim pretrained)
       ↓
  Fused Representation → Task Heads
  ```
  - Training: 50 epochs, batch 64, lr 2e-4
  - Expected: 15-25% improvement on electronic properties

- **exp202 (Concat Fusion)**:
  ```
  [Composition | Structure | DeePAW] → 384-dim
       ↓
  Linear projection → 128-dim
       ↓
  Task Heads
  ```
  - Tests if concatenation preserves more information than addition

- **exp203 (Angles + DeePAW)**:
  ```
  Structure Embedding (with 3-body angle features)
       +
  DeePAW Features
       ↓
  Task Heads
  ```
  - Adds angle features: captures bond angles for better geometry understanding
  - Tests synergy between geometric features and pretrained embeddings

- **exp204a (Long Training)** and **exp204b (Replace Mode)** - Special case with two variants:

  **exp204a (exp204_deepaw_long)**:
  ```
  Same as exp201 (Add fusion) but trained for 100 epochs
  ```
  - Training: 100 epochs (vs 50 in exp201)
  - Tests if longer training improves convergence
  - Expected: Performance plateau around epoch 70-80

  **exp204b (exp204_deepaw_replace)**:
  ```
  DeePAW Features (replace composition embedding entirely)
       +
  Structure Embedding (from GNN)
       ↓
  Task Heads
  ```
  - Tests if DeePAW alone is sufficient, no separate composition embedding
  - Hypothesis: Pretrained features already encode composition information
  - More parameter-efficient: reduces 3-16% parameters vs Add/Concat
  - Training: 50 epochs, batch 64, lr 1e-4

- **exp205 (Lower LR)**:
  - Same as exp201 but lr=1e-4 (instead of 2e-4)
  - Tests training stability with more conservative learning rate

- **exp207 (Two-Stage Fine-tuning)**:
  - Stage 1: Train backbone on all tasks (50 epochs)
  - Stage 2: Freeze backbone, fine-tune task heads only (20 epochs)
  - Tests if specialized head training improves performance

### Stage B Experiments (18 tasks including elastic)

**What it trains**: A single model that simultaneously predicts 18 materials properties:
1. **All 8 Stage A tasks** (thermodynamic + electronic)
2. **Structural** (2 tasks): volume, density
3. **Stability** (1 task): is_stable (classification)
4. **Elastic** (7 tasks): bulk_modulus_vrh, shear_modulus_vrh, youngs_modulus, homogeneous_poisson, universal_anisotropy, and 2 more

**Challenge**: Only ~11k materials (7%) have elastic data, while ~155k have electronic data. This creates severe data imbalance.

**Solution**: Weighted sampling with `--oversample-elastic N` to sample elastic materials N× more frequently during training.

**Architecture**: Shared backbone → 5 task-specific heads (thermodynamic + electronic + structural + stability + elastic)

**Experiment Series**:

#### Phase 3 Baseline (exp101-103)
- **exp101 (Graph Baseline v1)**:
  ```
  Input: Crystal Structure (155k materials, 11k with elastic)
       ↓
  Backbone: GNN (6.0Å cutoff, 24 neighbors, 64 RBF)
       ↓
  Shared Representation (256-dim)
       ↓
  5 Task Heads:
    ├─ Thermodynamic (3 outputs)
    ├─ Electronic (5 outputs)
    ├─ Structural (2 outputs)
    ├─ Stability (1 output)
    └─ Elastic (7 outputs)
  ```
  - Training: 50 epochs, batch 64, lr 1e-4, **no AMP** (dtype issues)
  - Weighted sampling: **4× oversampling** for elastic materials
  - Excluded: cbm_up, cbm_dn, vbm_up, vbm_dn (spin-polarized, low coverage)
  - Result:
    - Best: is_metal (AUROC 0.9575), band_gap (MAE 0.23 eV)
    - Challenge: Elastic properties show overfitting (53% train-val gap)

- **exp102 (Balanced v2)**:
  - Same as exp101 but **2× oversampling** (reduced from 4×)
  - Hypothesis: Lower oversampling may reduce elastic overfitting
  - Goal: Recover Stage A core task performance

- **exp103 (Core Guard v3)**:
  - Same as exp101 but **1× oversampling** (no oversampling)
  - Hypothesis: Prioritize core tasks, accept weaker elastic performance
  - Goal: Match Stage A performance on core 8 tasks

#### Phase 4 Single-Task Heads (exp104)
- **exp104 (Single-Task Fine-tuning)**:
  ```
  Stage 1: Train shared backbone on all 18 tasks
       ↓
  Stage 2: Freeze backbone, fine-tune each task head independently
       ↓
  Result: 18 separate models (shared backbone, task-specific heads)
  ```
  - Base checkpoint: exp103 best model
  - Training: One job per task, isolated fine-tuning
  - Hypothesis: Reduce cross-task interference during head training
  - Output: `artifacts/runs_stageb_v4/<task>/<run_id>/`

#### Phase 3 PyG Backend (exp105)
- **exp105 (Stage A PyG Baseline)**:
  - Retrain Stage A baseline (8 tasks) with PyTorch Geometric backend
  - Same architecture as exp001, but using PyG data structures
  - Purpose: Validate PyG backend before using for Stage B
  - Output: `artifacts/runs_stagea_pyg/`

#### Phase 2 DeePAW (exp206)
- **exp206 (DeePAW for Stage B)**:
  - Same DeePAW integration as Stage A, but with all 18 tasks
  - Tests if pretrained features help elastic property prediction
  - Hypothesis: Charge density features may correlate with mechanical properties

### Stage C Experiments (Head Architecture Variants)

**Concept**: Specialized head architectures for better task relationships.

- **exp107 (Electronic Hierarchical Head)**:
  ```
  Backbone → Electronic Group Head → Task Heads (band_gap, cbm, vbm, efermi)
  ```
  - Hierarchy: Learn shared electronic representation, then specialize
  - Hypothesis: Band structure tasks share common features

- **exp108 (Elastic Derived Head)**:
  ```
  Backbone → Base Moduli Head (bulk, shear) → Derived Heads (Poisson, anisotropy)
  ```
  - Physics-informed: Poisson ratio and anisotropy derived from moduli
  - Hypothesis: Enforce physical relationships in architecture

- **exp109 (Hybrid Hierarchical)**:
  - Combines exp107 + exp108
  - Electronic hierarchy + Elastic derived hierarchy
  - Tests if both hierarchies improve overall performance

## Standard Experiment Structure

Each experiment directory contains:

```
expXXX_name/
├── README.md                 # Experiment documentation
│                             # - Hypothesis and motivation
│                             # - Architecture details
│                             # - Expected outcomes
├── train.sh                  # Training script (executable)
│                             # - Complete training command
│                             # - All hyperparameters documented
├── config.json              # Saved training configuration
│                             # (copied from artifacts after training)
├── training_log.txt         # Console output
│                             # (copied from artifacts after training)
└── analysis/                # Post-training analysis (optional)
    ├── performance_report.md
    ├── visualization.png
    └── predictions.json
```

**Actual training outputs** are saved to `artifacts/runs_exp{ID}/`:
```
artifacts/runs_exp{ID}/
├── config.json                    # Complete training configuration
├── checkpoints/
│   ├── best.pt                   # Best validation checkpoint
│   └── last.pt                   # Latest checkpoint
├── metrics/
│   ├── train_metrics.json        # Per-epoch training metrics
│   ├── val_metrics.json          # Per-epoch validation metrics
│   └── best_summary.json         # Best performance summary
└── logs/
    └── training.log              # Full console output
```

## Quick Reference

### Understanding Experiment IDs

| Experiment | Stage | Tasks | Backbone | Key Feature | Purpose |
|------------|-------|-------|----------|-------------|---------|
| exp001 | A | 8 | Composition | Element embedding only | Baseline without structure |
| exp002 | A | 8 | Graph | GNN (6.0Å, 24 neighbors) | Baseline with structure |
| exp101 | B | 18 | Graph | GNN + 4× elastic oversampling | Baseline with all tasks |
| exp102 | B | 18 | Graph | GNN + 2× elastic oversampling | Balanced oversampling |
| exp103 | B | 18 | Graph | GNN + 1× (no oversampling) | Core task priority |
| exp104 | B | 18 | Graph | Single-task fine-tuning | Reduce task interference |
| exp105 | A | 8 | Graph (PyG) | PyG backend validation | Backend migration test |
| exp106 | Zero | 15 | Graph (per-task) | No multitask, adaptive capacity | Single-task upper bound |
| exp107 | C | 18 | Graph | Electronic hierarchical head | Task hierarchy |
| exp108 | C | 18 | Graph | Elastic derived head | Physics-informed |
| exp109 | C | 18 | Graph | Hybrid hierarchical | Combined hierarchies |
| exp201 | A | 8 | Enhanced + DeePAW | Add fusion | Test pretrained features |
| exp202 | A | 8 | Enhanced + DeePAW | Concat fusion | Alternative fusion strategy |
| exp203 | A | 8 | Enhanced + DeePAW | Angles + DeePAW | Geometric + pretrained |
| exp204a | A | 8 | Enhanced + DeePAW | Long training (100 epochs) | Extended training time |
| exp204b | A | 8 | Enhanced + DeePAW | Replace mode | DeePAW replaces composition |
| exp205 | A | 8 | Enhanced + DeePAW | Lower LR (1e-4) | Stability test |
| exp207 | A | 8 | Enhanced + DeePAW | Two-stage training | Fine-tuning ablation |
| exp206 | B | 18 | Enhanced + DeePAW | DeePAW for Stage B | Pretrained + elastic |

### Next Experiment ID to Use

- **Stage A experiments**: Use exp208+ (continue DeePAW series)
  - Note: exp204 has two variants (204a: long, 204b: replace), next ID is 208
- **Stage B experiments**: Use exp110+ (new enhancements)
- **Stage C experiments**: Use exp110+ (head variants)
- **Zero version**: exp106 complete (single-task baseline established)
- **New experiment series**: Use exp301+ range

### Backbone Comparison

| Backbone | Cutoff | Neighbors | RBF | Angles | DeePAW | Use Case |
|----------|--------|-----------|-----|--------|--------|----------|
| composition | N/A | N/A | N/A | ❌ | ❌ | Element features only |
| graph | 6.0Å | 24 | 64 | ❌ | ❌ | Standard GNN baseline |
| enhanced_graph | 8.0Å | 48 | 128 | ✅ | ❌ | Improved GNN |
| enhanced_graph + DeePAW | 8.0Å | 48 | 128 | ✅ | ✅ | GNN + pretrained |

### Common Training Patterns

**Pattern 1: Stage A + DeePAW Add Fusion**
```bash
python scripts/train_multitask.py \
  --split data/splits/split_iid_seed42.json \
  --stage a \
  --backbone enhanced_graph \
  --cutoff 8.0 --max-neighbors 48 --n-rbf 128 \
  --use-deepaw --deepaw-fusion add \
  --exclude-tasks volume density is_stable \
  --batch-size 64 --lr 2e-4 --epochs 50 \
  --out-dir artifacts/runs_exp201
```
**Trains**: 1 model → 8 tasks (thermodynamic + electronic)
**Architecture**: GNN + DeePAW → Shared features → 2 task heads

**Pattern 2: Stage A + DeePAW Replace Mode (exp204b)**
```bash
python scripts/train_multitask.py \
  --split data/splits/split_iid_seed42.json \
  --stage a \
  --backbone enhanced_graph \
  --cutoff 8.0 --max-neighbors 48 --n-rbf 128 \
  --use-deepaw --deepaw-fusion replace \
  --exclude-tasks volume density is_stable \
  --batch-size 64 --lr 2e-4 --epochs 50 \
  --out-dir artifacts/runs_exp204
```
**Trains**: 1 model → 8 tasks
**Architecture**: GNN + DeePAW (no composition embedding) → Shared features → 2 task heads
**Key**: More parameter-efficient, DeePAW replaces composition embedding entirely

**Pattern 3: Stage B + Elastic Oversampling**
```bash
python scripts/train_multitask.py \
  --split data/splits/split_iid_seed42.json \
  --stage b \
  --backbone graph \
  --batch-size 64 --lr 1e-4 --epochs 50 \
  --oversample-elastic 4.0 --no-amp \
  --exclude-tasks cbm_up cbm_dn vbm_up vbm_dn \
  --out-dir artifacts/runs_exp101
```
**Trains**: 1 model → 18 tasks (all properties including elastic)
**Architecture**: GNN → Shared features → 5 task heads
**Key**: 4× oversampling for elastic materials (7% coverage)

## Training Configuration Details

### Stage A Training Command
```bash
python scripts/train_multitask.py \
  --split data/splits/split_iid_seed42.json \
  --stage a \
  --backbone enhanced_graph \
  --cutoff 8.0 \              # Larger neighborhood (vs 6.0Å baseline)
  --max-neighbors 48 \        # More neighbors per atom (vs 24 baseline)
  --n-rbf 128 \               # More RBF basis functions (vs 64 baseline)
  --use-deepaw \              # Enable DeePAW pretrained features
  --deepaw-fusion add \       # Fusion strategy: add/concat/replace
  --exclude-tasks volume density is_stable \  # CRITICAL: prevents NaN loss
  --batch-size 64 \
  --lr 2e-4 \
  --epochs 50 \
  --out-dir artifacts/runs_exp201
```

**What this trains**:
- **Input**: 155k crystal structures from Materials Project
- **Backbone**: Enhanced GNN with DeePAW features
  - Message passing on atomic graph (8.0Å cutoff)
  - Pretrained atomic embeddings from 1.8M DFT calculations
  - 128-dim shared representation
- **Output**: 8 simultaneous predictions per material
  - 3 thermodynamic properties (regression)
  - 4 electronic properties (regression)
  - 1 classification (is_metal)
- **Loss**: Weighted sum of 8 task losses (weights based on data coverage)
- **Optimization**: AdamW with gradient clipping (max_norm=1.0)

### Stage B Training Command
```bash
python scripts/train_multitask.py \
  --split data/splits/split_iid_seed42.json \
  --stage b \
  --backbone graph \
  --batch-size 64 \
  --lr 1e-4 \                 # Lower LR for stability
  --epochs 50 \
  --oversample-elastic 4.0 \  # CRITICAL: balance elastic data (7% coverage)
  --no-amp \                  # Disable AMP (dtype issues with elastic tasks)
  --exclude-tasks cbm_up cbm_dn vbm_up vbm_dn \  # Low coverage, unstable
  --out-dir artifacts/runs_exp101
```

**What this trains**:
- **Input**: Same 155k materials, but only 11k have elastic properties
- **Backbone**: Standard GNN (6.0Å cutoff, 24 neighbors, 64 RBF)
- **Output**: 18 simultaneous predictions per material
  - All 8 Stage A tasks
  - 2 structural properties
  - 1 stability classification
  - 7 elastic/mechanical properties
- **Sampling**: WeightedRandomSampler
  - Materials with elastic data: sampled 4× more frequently
  - Balances training signal across high/low coverage tasks
- **Loss**: Weighted sum of 18 task losses
  - Higher weights for low-coverage tasks (elastic)
  - Prevents high-coverage tasks from dominating gradient

## Documentation

- **This file**: Experiment naming, architecture, training details
- **Stage A Summary**: [stage_a/summary.md](stage_a/summary.md) - Phase 1 & 2 results
- **Stage B Summary**: [stage_b/summary.md](stage_b/summary.md) - Phase 3 results
- **Phase 1 Complete Report**: [../reports/PHASE1_COMPLETE_REPORT.md](../reports/PHASE1_COMPLETE_REPORT.md)
- **Phase 2 Training Plan**: [../reports/PHASE2_TRAINING_PLAN.md](../reports/PHASE2_TRAINING_PLAN.md)
- **Academic Paper**: [../reports/academic_paper.md](../reports/academic_paper.md)
- **Experiment Tracking**: [EXPERIMENTS.md](EXPERIMENTS.md) - Master tracking table

## Key Concepts

### Multitask Learning
- **One model, multiple tasks**: All tasks share the same backbone (feature extractor)
- **Joint training**: Tasks learn together, helping each other through shared representations
- **Task heads**: Separate output layers for each task group (thermodynamic, electronic, etc.)
- **Weighted loss**: Each task contributes to the total loss based on data coverage

### Data Imbalance (Stage B)
- **Problem**: 155k materials have electronic data, only 11k have elastic data
- **Solution**: Weighted sampling - sample elastic materials 4× more frequently
- **Effect**: Balances gradient contributions from high/low coverage tasks

### DeePAW Integration
- **What**: Pretrained atomic embeddings from 1.8M DFT charge density calculations
- **Why**: Charge density encodes electronic structure information
- **How**: Fuse with composition/structure embeddings (add/concat/replace)
- **Expected**: 15-25% improvement on electronic properties (band_gap, cbm, vbm)

### Training Stability
- **Stage A**: Can use AMP, lr=2e-4, must exclude volume/density/is_stable
- **Stage B**: Must use --no-amp (dtype issues), lr=1e-4, must oversample elastic
- **Gradient clipping**: max_norm=1.0 prevents gradient explosion
- **Warmup**: 5 epochs of learning rate warmup for stable initialization

## Best Practices

1. **Always document**: Update experiment README.md with hypothesis, architecture, and results
2. **Track progress**: Update EXPERIMENTS.md master table after each run
3. **Isolate outputs**: Use `--out-dir artifacts/runs_exp{ID}` for each experiment
4. **Save checkpoints**: Copy best.pt and config.json to experiment directory after training
5. **Run analysis**: Generate performance reports and visualizations
6. **Compare results**: Document improvements vs baseline with specific metrics
7. **Stage awareness**: Clearly indicate Stage A (8 tasks) vs Stage B (18 tasks)
8. **Understand architecture**: Know what your model is learning (shared backbone → task heads)

## Current Status

- ✅ **Phase 1 (exp01-02)**: Stage A baseline complete
  - exp01: Composition baseline (won 9/11 tasks, strong baseline)
  - exp02: Graph baseline (won 2/11 tasks, underperformed)
  - **Key finding**: Composition features dominate most tasks
- ✅ **Phase 3 (exp101-105)**: Stage B baseline and variants complete
  - exp101: Graph baseline with all 18 tasks (4× oversampling)
    - **Best**: is_metal (AUROC 0.9575), band_gap (MAE 0.23 eV)
    - **Challenge**: 53% train-val gap on elastic properties
  - exp102: Balanced (2× oversampling) - tested lower oversampling
  - exp103: Core guard (1× oversampling) - prioritize core tasks
  - exp104: Single-task fine-tuning - reduce task interference
  - exp105: PyG backend validation for Stage A
- ✅ **Zero Version (exp106)**: Single-task baseline complete
  - 15 independent models, adaptive capacity (160-320 dim)
  - **Purpose**: Upper bound for multitask comparison
- 📋 **Stage C (exp107-109)**: Head architecture variants planned
  - exp107: Electronic hierarchical head
  - exp108: Elastic derived head (physics-informed)
  - exp109: Hybrid hierarchical (combined)
- 🚀 **Phase 2 (exp201-207)**: DeePAW integration in progress
  - exp201-207: Systematic DeePAW experiments on Stage A
  - **Special case**: exp204 has two variants
    - exp204a (exp204_deepaw_long): 100 epochs training
    - exp204b (exp204_deepaw_replace): Replace fusion mode
  - **Goal**: 15-25% improvement on electronic properties
  - **Status**: Training exp201 (Add fusion)
- 📋 **Future (exp206)**: DeePAW for Stage B (planned)
  - Test if pretrained features help elastic property prediction

See [EXPERIMENTS.md](EXPERIMENTS.md) for detailed tracking and results.
