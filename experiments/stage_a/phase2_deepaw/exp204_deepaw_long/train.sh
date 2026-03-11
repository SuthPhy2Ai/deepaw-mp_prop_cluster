#!/bin/bash
# EXP-204: DeePAW Long Training (100 epochs)
# 目标: 验证更长训练时间是否能进一步降低 MAE
# 预期: 验证集性能在 70-80 epoch 后趋于稳定

set -e

echo "========================================="
echo "EXP-204: DeePAW Long Training (100 epochs)"
echo "========================================="
echo "Backbone: enhanced_graph"
echo "DeePAW Fusion: add"
echo "Epochs: 100"
echo "Learning Rate: 2e-4"
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
  --epochs 100 \
  --lr 2e-4 \
  --weight-decay 1e-5 \
  --grad-clip 1.0 \
  --warmup-epochs 5 \
  --no-amp \
  --out-dir artifacts/runs_exp204

echo "========================================="
echo "Training completed!"
echo "Results saved to artifacts/runs_exp204"
echo "========================================="
