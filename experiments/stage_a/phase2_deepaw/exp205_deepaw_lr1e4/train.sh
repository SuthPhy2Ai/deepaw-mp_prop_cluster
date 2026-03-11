#!/bin/bash
# EXP-205: DeePAW with Lower Learning Rate
# 目标: 验证较低学习率是否能提高训练稳定性
# 预期: 训练更稳定，可能收敛更慢但最终性能相近

set -e

echo "========================================="
echo "EXP-205: DeePAW Lower LR Training"
echo "========================================="
echo "Backbone: enhanced_graph"
echo "DeePAW Fusion: add"
echo "Epochs: 50"
echo "Learning Rate: 1e-4 (lower)"
echo "Batch Size: 64"
echo "========================================="

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
  --lr 1e-4 \
  --weight-decay 1e-5 \
  --grad-clip 1.0 \
  --warmup-epochs 5 \
  --no-amp \
  --out-dir artifacts/runs_exp205

echo "========================================="
echo "Training completed!"
echo "Results saved to artifacts/runs_exp205"
echo "========================================="
