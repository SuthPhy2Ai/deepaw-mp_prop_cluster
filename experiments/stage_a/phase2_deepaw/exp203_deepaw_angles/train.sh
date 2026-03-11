#!/bin/bash
# EXP-203: DeePAW + Enhanced Graph + Angles
# 目标: 验证 DeePAW 特征与三体角度特征的协同效果
# 预期: 结构性质可能进一步改进

set -e

echo "========================================="
echo "EXP-203: DeePAW + Angles Training"
echo "========================================="
echo "Backbone: enhanced_graph"
echo "DeePAW Fusion: add"
echo "Use Angles: True"
echo "Epochs: 50"
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
  --use-angles \
  --use-deepaw-features \
  --deepaw-checkpoint /home/sutianhao/data/deepaw_test/DeePAW-main/checkpoints/f_nonlocal_escn_best.pth \
  --deepaw-fusion add \
  --batch-size 64 \
  --num-workers 8 \
  --exclude-tasks volume density is_stable \
  --epochs 50 \
  --lr 2e-4 \
  --weight-decay 1e-5 \
  --grad-clip 1.0 \
  --warmup-epochs 5 \
  --no-amp \
  --out-dir artifacts/runs_exp203

echo "========================================="
echo "Training completed!"
echo "Results saved to artifacts/runs_exp203"
echo "========================================="
