# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Materials Project (MP) data pipeline with two main capabilities:

1. **Data Pipeline**: Downloads materials data from Materials Project API or loads from pre-downloaded JSONL.gz file, stores in ASE (Atomic Simulation Environment) SQLite database
2. **Multitask ML Training**: Graph neural network models for predicting crystal properties (thermodynamic, electronic, structural, elastic) with support for multiple data split strategies

**Pre-downloaded Dataset**: `data/raw/summary_all_merged.jsonl.gz` (280MB compressed) contains ~155k materials with complete structural, energetic, electronic, and elastic properties. Use this to avoid API calls.

**Current Status**:
- ✅ Phase 1 (Stage A baseline): Completed (see [docs/project_status/PHASE1_DONE.txt](docs/project_status/PHASE1_DONE.txt))
- ✅ Phase 3 (Stage B baseline): Completed - exp101_baseline_graph trained on all 18 tasks
- ✅ Academic paper: Comprehensive research paper written (see [reports/academic_paper.md](reports/academic_paper.md))
- 🚀 Phase 2 enhanced graph backbone implementation ready for experimentation

## Architecture

### Data Pipeline (3 stages)

1. **Data Loading** ([load_from_jsonl.py](scripts/load_from_jsonl.py)): Reads JSONL.gz and writes to ASE database (recommended path)
   - Alternative: [fetch_mp_data.py](scripts/fetch_mp_data.py) downloads from API with checkpoint-based resumption
2. **Database Storage**: ASE SQLite with two-tier storage:
   - `key_value_pairs`: Searchable scalars (energy, band gap, moduli)
   - `data` dict: Complex arrays (6×6 elastic tensors)
3. **Validation** ([validate.py](scripts/validate.py)): Completeness and physical sanity checks

### Multitask ML Pipeline (4 stages)

1. **Split Generation** ([export_splits.py](scripts/export_splits.py)): Creates train/val/test splits
   - IID: Random 80/10/10 split
   - ChemSys-OOD: Grouped by chemical system (~70/15/15)
   - Complexity-OOD: By element count (train ≤4, val=5, test ≥6)
2. **Training** ([train_multitask.py](scripts/train_multitask.py)): Multi-stage training
   - Stage A: High-coverage tasks (excludes elastic properties)
   - Stage B/C: All tasks including elastic (uses weighted sampling to oversample elastic data)
3. **Evaluation** ([eval_multitask.py](scripts/eval_multitask.py)): Checkpoint evaluation
4. **Model Architecture**:
   - Backbones: Graph (SchNet-style message passing), Composition (element embedding), or EnhancedGraph (with angle features and improved architecture)
   - Grouped task heads: Separate heads for thermodynamic, electronic, structural, elastic properties
   - Task-specific loss weighting based on data coverage

### Key Design Patterns

- **Two-tier ASE storage**: Searchable scalars in `key_value_pairs`, complex data in `data` dict
- **Pymatgen ↔ ASE conversion**: Fast-path for ordered structures, fallback to pymatgen for disordered
- **Schema quirks**: `formula_pretty` (not `formula`), `pg_`-prefixed point groups to avoid ASE string/int ambiguity
- **Staged training**: Stage A trains on high-coverage tasks, Stage B adds elastic properties with oversampling
- **Weighted sampling**: Stage B/C oversample materials with elastic data (default 4×) to balance low-coverage tasks
- **Progressive enhancement**: EnhancedGraph backbone supports incremental feature activation (basic → angles → edge updates) for ablation studies

## Common Commands

### Setup
```bash
pip install -r requirements.txt

# Only needed for API download (not required for pre-downloaded data)
export MP_API_KEY="your_materials_project_api_key"
```

### Testing
```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_enhanced_backbones.py

# Run with verbose output
python -m pytest tests/ -v
```

### Data Pipeline (Recommended: Use Pre-downloaded Data)
```bash
# Build ASE database from JSONL.gz (fast, no API needed)
python scripts/load_from_jsonl.py --overwrite

# Validate database
python scripts/validate.py

# Alternative: Download from API (slow, requires API key)
python scripts/fetch_mp_data.py  # Downloads with checkpoints
python scripts/store_to_ase.py   # Converts checkpoints to ASE DB
```

### Multitask ML Training
```bash
# 1. Generate data splits (IID, ChemSys-OOD, Complexity-OOD)
python scripts/export_splits.py

# 2. Train Stage A (high-coverage tasks, no elastic)
# CRITICAL: Must exclude volume/density/is_stable to prevent NaN loss
python scripts/train_multitask.py \
  --split data/splits/split_iid_seed42.json \
  --stage a \
  --backbone graph \
  --batch-size 64 \
  --num-workers 8 \
  --exclude-tasks volume density is_stable \
  --epochs 50

# 3. Train Stage B (all tasks including elastic, with oversampling)
# IMPORTANT: Use --no-amp and lower learning rate for stability
# Exclude spin-polarized CBM/VBM tasks (low coverage, unstable)
python scripts/train_multitask.py \
  --split data/splits/split_iid_seed42.json \
  --stage b \
  --backbone graph \
  --batch-size 64 \
  --num-workers 8 \
  --epochs 50 \
  --oversample-elastic 4.0 \
  --no-amp \
  --lr 1e-4 \
  --exclude-tasks cbm_up cbm_dn vbm_up vbm_dn

# 4. Train with Enhanced Graph Backbone (Phase 2)
# Basic enhancements (larger cutoff, more RBF basis functions)
python scripts/train_multitask.py \
  --split data/splits/split_iid_seed42.json \
  --stage a \
  --backbone enhanced_graph \
  --cutoff 8.0 \
  --max-neighbors 48 \
  --n-rbf 128 \
  --lr 2e-4 \
  --epochs 50

# With angle features (three-body interactions)
python scripts/train_multitask.py \
  --split data/splits/split_iid_seed42.json \
  --stage a \
  --backbone enhanced_graph \
  --cutoff 8.0 \
  --max-neighbors 48 \
  --n-rbf 128 \
  --use-angles \
  --lr 2e-4 \
  --epochs 50

# Full enhancements (angles + edge updates)
python scripts/train_multitask.py \
  --split data/splits/split_iid_seed42.json \
  --stage a \
  --backbone enhanced_graph \
  --cutoff 8.0 \
  --max-neighbors 48 \
  --n-rbf 128 \
  --use-angles \
  --use-edge-update \
  --lr 2e-4 \
  --epochs 50

# 5. Evaluate checkpoint
python scripts/eval_multitask.py \
  --split data/splits/split_iid_seed42.json \
  --checkpoint artifacts/runs/<run_id>/checkpoints/best.pt

# 6. Generate comprehensive evaluation report (Claude-style)
# This creates a detailed markdown report with:
# - Performance tables for all tasks
# - 6 evaluation figures (ROC curves, scatter plots, training curves)
# - Statistical analysis and recommendations
python scripts/generate_gpt_eval_report.py \
  --run-dir artifacts/runs/<run_id> \
  --output-dir experiments/stage_b/phase3_baseline/exp101_baseline_graph/claude_evaluation
```

### Academic Paper Generation
After completing experiments, generate research paper documentation:

```bash
# The academic paper documents:
# - Complete methodology (architecture, training, data handling)
# - Comprehensive results (18 tasks with metrics)
# - Analysis and discussion (strengths, limitations, future work)
# - Supplementary materials (hyperparameters, implementation details)

# Paper location: reports/academic_paper.md
# Supplementary: reports/academic_paper_supplementary.md
# Figures: reports/academic_paper_figures/ (6 evaluation plots)
```

### Querying ASE Database
```python
from ase.db import connect

db = connect('data/db/mp_materials.db')
print(db.count())

# Query by properties (note: use formula_pretty, not formula)
for row in db.select('band_gap>2.0, is_stable=True', limit=10):
    print(row.mp_id, row.formula_pretty, row.band_gap)

# Get structure as ASE Atoms
row = db.get(mp_id='mp-149')
atoms = row.toatoms()

# Access complex data (elastic tensors, moduli breakdowns)
print(row.data.get('elastic_tensor_voigt'))
print(row.data.get('bulk_modulus'))  # dict with vrh/voigt/reuss
```

## Configuration

All configuration is in [src/mp_data_pipeline/config.py](src/mp_data_pipeline/config.py):
- `MP_API_KEY`: Read from environment variable (only for API download)
- `CHUNK_SIZE`: Materials per API request (default: 500)
- `MAX_RETRIES`: Retry attempts for failed API calls (default: 3)
- `DB_PATH`: Output database (`data/db/mp_materials.db`)
- `RAW_JSONL_PATH`: Pre-downloaded data (`data/raw/summary_all_merged.jsonl.gz`)
- `CHECKPOINT_DIR`: API download checkpoints (`data/checkpoints/`)

## Data Schema

### ASE Database Key-Value Pairs (searchable)
- `mp_id`: Materials Project ID (string)
- `formula_pretty`: Chemical formula (string) - **Note: use `formula_pretty`, not `formula`**
- `crystal_system`, `point_group`: Symmetry (string, point_group prefixed with `pg_`)
- `spacegroup`, `nsites`: Integer fields
- `is_stable`, `is_metal`: Boolean flags
- `energy_per_atom`, `formation_energy_per_atom`, `energy_above_hull`: Thermodynamics (float)
- `band_gap`, `cbm`, `vbm`, `efermi`: Electronic properties (float)
- `volume`, `density`: Structural properties (float)
- `bulk_modulus_vrh`, `shear_modulus_vrh`, `youngs_modulus`, `homogeneous_poisson`, `universal_anisotropy`: Elastic moduli (float)

### ASE Database Data Dict (non-searchable)
- `elastic_tensor_voigt`: 6×6 elastic tensor (list)
- `compliance_tensor_voigt`: 6×6 compliance tensor (list)
- `bulk_modulus`, `shear_modulus`: Dicts with `vrh`, `voigt`, `reuss` breakdown

### Multitask ML Tasks (18 total)
Grouped into 5 categories:
- **Thermodynamic** (3): energy_per_atom, formation_energy_per_atom, energy_above_hull
- **Electronic** (5): band_gap, cbm, vbm, efermi (regression), is_metal (classification)
- **Stability** (1): is_stable (classification)
- **Structural** (2): volume, density
- **Elastic** (4): bulk_modulus_vrh, shear_modulus_vrh, homogeneous_poisson, universal_anisotropy

Stage A trains on first 14 tasks (excludes elastic). Stage B/C trains on all 18 tasks.

## Dataset Statistics

Pre-downloaded dataset (`data/raw/summary_all_merged.jsonl.gz`):
- **154,879 materials** total
- **10,994 materials (7.1%)** with elasticity data
- **33,973 materials (21.9%)** thermodynamically stable
- **72,640 materials (46.9%)** metallic

## Important Notes

### Data Pipeline
- **Recommended Path**: Use `load_from_jsonl.py` with pre-downloaded data (no API key needed)
- **API Key**: Only required for `fetch_mp_data.py` (alternative download path)
- **Checkpoint Resume**: `fetch_mp_data.py` resumes from last checkpoint if interrupted
- **Schema Quirks**:
  - Use `formula_pretty` field, not `formula`
  - Point groups stored as `pg_-1`, `pg_1`, etc. to avoid ASE string/int ambiguity
  - Elastic properties only available for ~7% of materials
- **Young's Modulus**: Computed from bulk/shear moduli using E = 9KG/(3K+G)
- **Database Locks**: If writes hang, check for stale `.db.lock` files and remove them
- **Database Queries**: Use `db.get(mp_id=...)` for direct lookups instead of iterating through all rows for better performance

### Multitask ML Training
- **Stage A vs B**: Stage A excludes elastic tasks (high coverage), Stage B includes all tasks
- **Weighted Sampling**: Stage B/C use `WeightedRandomSampler` to oversample elastic data (default 4×)
- **Data Splits**: Three strategies available (IID, ChemSys-OOD, Complexity-OOD)
- **Task Weights**: Automatically computed based on data coverage in training set
- **Band Gap Constraint**: Model enforces non-negative band gap via softplus activation
- **Artifacts**: Training outputs saved to `artifacts/runs/<timestamp>/` with config, checkpoints, metrics
- **Lazy Loading**: Dataset uses lazy loading with direct database queries (`db.get(mp_id=...)`) to avoid loading all samples into memory during initialization
- **Mask Computation**: Training script computes task masks in parallel using multiprocessing to speed up initialization
- **Stage B Training Success** (exp101):
  - Disabled AMP due to dtype issues (use `--no-amp`)
  - Reduced learning rate to 1e-4 for stability
  - 4× elastic oversampling successfully balanced low-coverage tasks
  - Excluded problematic tasks: `cbm_up`, `cbm_dn`, `vbm_up`, `vbm_dn` (spin-polarized, low coverage)
  - Training time: ~33 minutes for 50 epochs on GPU
- **Backbone Options**:
  - `graph`: Original SchNet-style message passing (6.0Å cutoff, 24 neighbors, 64 RBF)
  - `composition`: Element embedding only (no structure)
  - `enhanced_graph`: Enhanced architecture with configurable features:
    - Larger graph coverage (8.0Å cutoff, 48 neighbors recommended)
    - More RBF basis functions (128 recommended)
    - Optional angle features (`--use-angles`) for three-body interactions
    - Optional edge update mechanism (`--use-edge-update`)
    - AMP-compatible (fixes dtype issues from Phase 1)

### Phase 1 Baseline Results (Stage A - 8 Core Tasks)
Phase 1 established baseline performance with two models:
- **EXP-01 (Composition)**: Strong baseline, won 9/11 tasks (82%)
  - is_metal AUROC: 0.9098, band_gap MAE: 0.715 eV, is_stable AUROC: 0.8510
- **EXP-02 (Graph)**: Underperformed expectations, won 2/11 tasks (18%)
  - Only better on volume (-41%) and energy_above_hull (-18%)
  - Required AMP disabled and lower learning rate due to training instability

Key finding: Simple composition features dominate most tasks, but graph structure shows potential for specific properties.

### Phase 3 Baseline Results (Stage B - All 18 Tasks)
**EXP-101 (Graph Baseline)**: Successfully trained on all 18 tasks including elastic properties
- **Training**: 50 epochs, 33 minutes total, 77.2% validation loss improvement
- **Best Performance**:
  - is_metal: AUROC 0.9575 (production-ready)
  - formation_energy_per_atom: MAE 0.0800 eV, R² 0.988
  - energy_above_hull: MAE 0.0644 eV, R² 0.924
  - band_gap: MAE 0.2308 eV, R² 0.916
- **Challenges**:
  - Elastic properties show moderate overfitting (7.1% data coverage)
  - Electronic properties (cbm, vbm, efermi) need improvement
  - Overall train-val gap: 53%
- **Key Achievements**:
  - Stable training with disabled AMP and reduced learning rate (1e-4)
  - Successful handling of data imbalance via 4× elastic oversampling
  - Physical constraints (non-negative band gap) enforced
  - Comprehensive evaluation with 6 figures (ROC curves, scatter plots, training curves)

**Academic Paper**: Complete research paper documenting methodology, results, and analysis available at [reports/academic_paper.md](reports/academic_paper.md) with supplementary materials and figures.

### Phase 2 Optimization Plan
Phase 2 focuses on improving training stability and model architecture to address Phase 1/3 limitations:
- **2A: Training Stability**: EMA, learning rate warmup, gradient clipping
- **2B: Graph Construction**: Per-atom distance-sorted neighbors instead of global limit
- **2C: Atomic Baselines**: Reference energy corrections for better energy predictions
- **2D: Advanced Heads**: Specialized tensor output layers for elastic properties
- **2E: E(3) Equivariance**: Integration of e3nn library for equivariant architectures

**Target Improvements** (vs Phase 3 baseline):
- Reduce train-val gap from 53% to <30%
- Improve electronic properties: cbm/vbm/efermi MAE by 20%+
- Better elastic property generalization with specialized heads
- Enable AMP training for 2-3× speedup

See [PHASE2_OPTIMIZATION_PLAN.md](reports/plans/PHASE2_OPTIMIZATION_PLAN.md) for detailed implementation roadmap.

## Project Structure

```
.
├── src/mp_data_pipeline/         # Core modules
│   ├── ml/                        # Dataset, splits, task definitions
│   │   ├── dataset.py            # Original graph dataset
│   │   ├── enhanced_dataset.py   # Enhanced dataset with angle features
│   │   ├── splits.py             # Data splitting logic
│   │   └── tasks.py              # Task definitions and grouping
│   ├── models/                    # Backbones, heads, multitask model
│   │   ├── backbones.py          # Original graph and composition backbones
│   │   ├── enhanced_backbones.py # Enhanced graph backbone (Phase 2)
│   │   ├── graph_features.py     # Angle computation and feature expansion
│   │   ├── heads.py              # Task-specific prediction heads
│   │   └── multitask_model.py    # Main multitask model
│   ├── training/                  # Trainer, losses, sampler
│   ├── fetch_mp_data.py          # API download logic
│   ├── load_from_jsonl.py        # JSONL.gz loader
│   ├── store_to_ase.py           # Checkpoint to ASE converter
│   └── validate.py               # Database validation
├── scripts/                       # CLI entry points
│   ├── load_from_jsonl.py        # Recommended: load pre-downloaded data
│   ├── export_splits.py          # Generate train/val/test splits
│   ├── train_multitask.py        # Train multitask models
│   ├── eval_multitask.py         # Evaluate checkpoints
│   └── legacy/                   # Historical scripts
├── tests/                         # Unit tests
│   └── test_enhanced_backbones.py # Tests for enhanced backbone
├── data/
│   ├── raw/                      # summary_all_merged.jsonl.gz (280MB)
│   ├── db/                       # mp_materials.db (ASE SQLite)
│   ├── splits/                   # Train/val/test split JSON files
│   └── checkpoints/              # API download checkpoints
├── artifacts/runs/               # Training outputs (config, checkpoints, metrics)
├── experiments/                  # Organized experiment results
│   ├── stage_a/phase1_baseline/  # Phase 1: 8 core tasks (EXP-01, EXP-02)
│   └── stage_b/phase3_baseline/  # Phase 3: All 18 tasks (EXP-101)
│       └── exp101_baseline_graph/
│           ├── claude_evaluation/  # 6 evaluation figures + report
│           ├── TRAINING_COMPLETE.md
│           └── training_log_*.txt
├── reports/                      # Analysis reports and visualizations
│   ├── academic_paper.md         # Complete research paper
│   ├── academic_paper_supplementary.md
│   ├── academic_paper_figures/   # Paper figures (6 evaluation plots)
│   ├── PHASE1_COMPLETE_REPORT.md # Phase 1 baseline comparison
│   ├── PHASE1_INDEX.md           # Documentation index
│   ├── figures/                  # Visualization charts
│   └── plans/                    # Research plans and experiment logs
└── logs/                         # Application logs
```
