#!/bin/bash
# EXP-206: DeePAW Stage B (All Tasks)
# 目标: 训练包含弹性性质的完整模型
# 预期: 电子性质保持改进，弹性性质改进有限

set -e

echo "========================================="
echo "EXP-206: DeePAW Stage B Training"
echo "========================================="
echo "Backbone: enhanced_graph"
echo "DeePAW Fusion: add"
echo "Stage: B (16 tasks including elastic)"
echo "Epochs: 100"
echo "Learning Rate: 2e-4"
echo "Batch Size: 64"
echo "Oversample Elastic: 2.0x"
echo "========================================="

python scripts/train_multitask.py \
  --split data/splits/split_iid_seed42.json \
  --stage b \
  --backbone enhanced_graph \
  --cutoff 8.0 \
  --max-neighbors 48 \
  --n-rbf 128 \
  --use-deepaw-features \
  --deepaw-checkpoint /home/sutianhao/data/deepaw_test/DeePAW-main/checkpoints/f_nonlocal_escn_best.pth \
  --deepaw-fusion add \
  --batch-size 64 \
  --num-workers 8 \
  --epochs 100 \
  --lr 2e-4 \
  --weight-decay 1e-5 \
  --grad-clip 1.0 \
  --warmup-epochs 5 \
  --oversample-elastic 2.0 \
  --no-amp \
  --out-dir artifacts/runs_exp206

echo "========================================="
echo "Training completed!"
echo "Results saved to artifacts/runs_exp206"
echo "========================================="
