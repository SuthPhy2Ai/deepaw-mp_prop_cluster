#!/bin/bash
# DeePAW Pretrained Atom Features Integration
# Experiment: exp201_deepaw_features

set -e

# Configuration
SPLIT="data/splits/split_iid_seed42.json"
STAGE="a"
BACKBONE="enhanced_graph"
CUTOFF=8.0
MAX_NEIGHBORS=48
N_RBF=128
BATCH_SIZE=64
NUM_WORKERS=8
EPOCHS=50
LR=2e-4

# DeePAW configuration
USE_DEEPAW="--use-deepaw-features"
DEEPAW_CHECKPOINT="/home/sutianhao/data/deepaw_test/DeePAW-main/checkpoints/f_nonlocal_escn_best.pth"
DEEPAW_FUSION="add"  # or "concat"

# Exclude problematic tasks
EXCLUDE_TASKS="volume density is_stable"

echo "========================================="
echo "Training with DeePAW Pretrained Features"
echo "========================================="
echo "Backbone: $BACKBONE"
echo "DeePAW Checkpoint: $DEEPAW_CHECKPOINT"
echo "Fusion Method: $DEEPAW_FUSION"
echo "Cutoff: $CUTOFF Å"
echo "Max Neighbors: $MAX_NEIGHBORS"
echo "Batch Size: $BATCH_SIZE"
echo "Learning Rate: $LR"
echo "Epochs: $EPOCHS"
echo "========================================="

python scripts/train_multitask.py \
  --split "$SPLIT" \
  --stage "$STAGE" \
  --backbone "$BACKBONE" \
  --cutoff "$CUTOFF" \
  --max-neighbors "$MAX_NEIGHBORS" \
  --n-rbf "$N_RBF" \
  $USE_DEEPAW \
  --deepaw-checkpoint "$DEEPAW_CHECKPOINT" \
  --deepaw-fusion "$DEEPAW_FUSION" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --exclude-tasks $EXCLUDE_TASKS \
  --epochs "$EPOCHS" \
  --lr "$LR" \
  --no-amp

echo "========================================="
echo "Training completed!"
echo "========================================="
