#!/bin/bash
# EXP-201: DeePAW + Enhanced Graph (Add Fusion)
# 目标: 验证 DeePAW 预训练特征对电子性质预测的改进效果
# 预期: 电子性质 MAE 降低 15-25%，band_gap < 0.60 eV

set -e

echo "========================================="
echo "EXP-201: DeePAW Add Fusion Training"
echo "========================================="
echo "Backbone: enhanced_graph"
echo "DeePAW Fusion: add"
echo "Epochs: 50"
echo "Learning Rate: 2e-4"
echo "Batch Size: 16"
echo "========================================="

python scripts/train_multitask.py \
  --split data/splits/split_iid_seed42.json \
  --stage a \
  --backbone enhanced_graph \
  --cutoff 6.0 \
  --max-neighbors 24 \
  --n-rbf 128 \
  --use-deepaw-features \
  --deepaw-checkpoint /home/sutianhao/data/deepaw_test/DeePAW-main/checkpoints/f_nonlocal_escn_best.pth \
  --deepaw-fusion add \
  --batch-size 16 \
  --num-workers 8 \
  --exclude-tasks volume density is_stable \
  --epochs 50 \
  --lr 2e-4 \
  --weight-decay 1e-5 \
  --grad-clip 1.0 \
  --warmup-epochs 5 \
  --no-amp \
  --no-pyg \
  --out-dir artifacts/runs_exp201

echo "========================================="
echo "Training completed!"
echo "Results saved to artifacts/runs_exp201"
echo "========================================="
