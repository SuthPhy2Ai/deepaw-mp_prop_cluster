# Phase 2: DeePAW Pretrained Atom Features Integration

## Overview

This experiment integrates pretrained atom features from the DeePAW project into the MP materials property prediction pipeline. DeePAW is trained on charge density prediction tasks, providing rich electronic structure information that should improve predictions for electronic properties.

## Implementation

### Architecture

```
Input Structure
    ↓
EnhancedGraphDataset (with positions)
    ↓
DeePAWAtomFeatureExtractor
    ├─ Load pretrained F_nonlocal_escn checkpoint
    ├─ Extract atom tower features (3200-dim)
    └─ Use MP's pre-built graph (no PBC reconstruction)
    ↓
Project to hidden_dim (256)
    ↓
Fuse with learnable atom embeddings (add/concat)
    ↓
Message Passing Layers
    ↓
Task-specific Heads
```

### Key Features

1. **Pretrained Features**: Uses DeePAW's eSCN atom tower (105M parameters)
2. **Graph Reuse**: Leverages MP's pre-built graph structure (cutoff=8.0Å)
3. **Fusion Methods**:
   - `add`: Additive fusion (default)
   - `concat`: Concatenative fusion with projection
4. **Frozen Weights**: DeePAW extractor is frozen during training

## Training Command

```bash
./train_deepaw.sh
```

Or manually:

```bash
python scripts/train_multitask.py \
  --split data/splits/split_iid_seed42.json \
  --stage a \
  --backbone enhanced_graph \
  --cutoff 8.0 \
  --max-neighbors 48 \
  --n-rbf 128 \
  --use-deepaw-features \
  --deepaw-checkpoint /home/sutianhao/data/deepaw_test/DeePAW-main/checkpoints/f_nonlocal_escn_best.pth \
  --deepaw-fusion add \
  --batch-size 64 \
  --num-workers 8 \
  --exclude-tasks volume density is_stable \
  --epochs 50 \
  --lr 2e-4 \
  --no-amp
```

## Expected Improvements

Based on DeePAW's charge density pretraining, we expect improvements in:

- ✅ **band_gap**: Direct electronic structure correlation
- ✅ **cbm/vbm**: Conduction/valence band positions
- ✅ **efermi**: Fermi energy level
- ✅ **is_metal**: Metallic classification
- ⚠️ **formation_energy**: Indirect correlation
- ⚠️ **elastic properties**: Weaker correlation

**Target Metrics**:
- Electronic properties MAE: ↓ 20%+
- Train-val gap: ↓ to <40%
- is_metal AUROC: ↑ 0.02+

## Files

- `train_deepaw.sh`: Training script
- `README.md`: This file
- Results will be saved to `artifacts/runs/<timestamp>/`

## Notes

- DeePAW checkpoint must be available at the specified path
- Training uses `--no-amp` for stability
- Positions are required in batch_dict (automatically added by EnhancedGraphDataset)
- Cell information is NOT needed (方案B implementation)
