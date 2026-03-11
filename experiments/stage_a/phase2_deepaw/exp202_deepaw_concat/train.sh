#!/bin/bash
# EXP-202: DeePAW + Enhanced Graph (Concat Fusion)
# 目标: 对比 concat 融合策略与 add 融合的效果
# 预期: 保留更多信息，可能在电子性质上略优于 add

set -e

echo "========================================="
echo "EXP-202: DeePAW Concat Fusion Training"
echo "========================================="
echo "Backbone: enhanced_graph"
echo "DeePAW Fusion: concat"
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
  --use-deepaw-features \
  --deepaw-checkpoint /home/sutianhao/data/deepaw_test/DeePAW-main/checkpoints/f_nonlocal_escn_best.pth \
  --deepaw-fusion concat \
  --batch-size 64 \
  --num-workers 8 \
  --exclude-tasks volume density is_stable \
  --epochs 50 \
  --lr 2e-4 \
  --weight-decay 1e-5 \
  --grad-clip 1.0 \
  --warmup-epochs 5 \
  --no-amp \
  --out-dir artifacts/runs_exp202

echo "========================================="
echo "Training completed!"
echo "Results saved to artifacts/runs_exp202"
echo "========================================="
