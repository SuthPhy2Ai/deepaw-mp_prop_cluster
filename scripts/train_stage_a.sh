#!/bin/bash
# Stage A Training Script - 8 Core Tasks
# CRITICAL: Always exclude volume, density, is_stable to prevent NaN loss

set -e  # Exit on error

# Configuration
SPLIT="data/splits/split_iid_seed42.json"
STAGE="a"
BACKBONE="graph"
BATCH_SIZE=64
NUM_WORKERS=8
EPOCHS=50
LR=0.0001  # Reduced from 0.001 to prevent gradient explosion
GRAD_CLIP=1.0  # Add gradient clipping for stability
WARMUP_EPOCHS=5  # Add learning rate warmup

# CRITICAL: These tasks MUST be excluded
EXCLUDE_TASKS="volume density is_stable"

# Log file
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/train_stage_a_${TIMESTAMP}.log"

echo "========================================="
echo "Starting Stage A Training"
echo "========================================="
echo "Split: $SPLIT"
echo "Backbone: $BACKBONE"
echo "Batch size: $BATCH_SIZE"
echo "Workers: $NUM_WORKERS"
echo "Epochs: $EPOCHS"
echo "Learning rate: $LR"
echo "Gradient clip: $GRAD_CLIP"
echo "Warmup epochs: $WARMUP_EPOCHS"
echo "Excluded tasks: $EXCLUDE_TASKS"
echo "Log file: $LOG_FILE"
echo "========================================="

# Run training
python scripts/train_multitask.py \
  --split "$SPLIT" \
  --stage "$STAGE" \
  --backbone "$BACKBONE" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --exclude-tasks $EXCLUDE_TASKS \
  --epochs "$EPOCHS" \
  --lr "$LR" \
  --grad-clip "$GRAD_CLIP" \
  --warmup-epochs "$WARMUP_EPOCHS" \
  2>&1 | tee "$LOG_FILE"

echo "========================================="
echo "Training completed. Log saved to: $LOG_FILE"
echo "========================================="
