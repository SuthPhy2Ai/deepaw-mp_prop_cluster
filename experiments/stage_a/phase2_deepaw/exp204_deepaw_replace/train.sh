#!/bin/bash
# EXP-204: DeePAW Replace Fusion
# 用 DeePAW 特征直接替换 atom embedding（不融合）

set -e

cd /scratch/sutianhao/data/mp-data-pipeline

# 设置环境变量（假设已在 ctgan 环境中）
export PYTHONNOUSERSITE=1

# 训练配置
python scripts/train_multitask.py \
  --db data/db/mp_materials.db \
  --split data/splits/split_iid_seed42.json \
  --stage a \
  --backbone enhanced_graph \
  --hidden-dim 256 \
  --layers 6 \
  --cutoff 6.0 \
  --max-neighbors 24 \
  --n-rbf 64 \
  --use-deepaw-features \
  --deepaw-checkpoint /home/sutianhao/data/deepaw_test/DeePAW-main/checkpoints/f_nonlocal_escn_best.pth \
  --deepaw-fusion replace \
  --batch-size 64 \
  --num-workers 8 \
  --exclude-tasks volume density is_stable \
  --epochs 50 \
  --lr 1e-4 \
  --weight-decay 1e-5 \
  --ema-decay 0.0 \
  --warmup-epochs 5 \
  --grad-clip 1.0 \
  --no-amp \
  --use-pyg \
  --out-dir artifacts/runs_exp204

echo "EXP-204 训练完成！"
