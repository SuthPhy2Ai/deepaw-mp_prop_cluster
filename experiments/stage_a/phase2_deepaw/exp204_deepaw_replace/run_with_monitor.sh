#!/bin/bash
# EXP-204: 启动训练并监控

set -e

cd /scratch/sutianhao/data/mp-data-pipeline

# 创建日志目录
mkdir -p logs

# 日志文件
TRAIN_LOG="logs/exp204_replace_train.log"
WATCH_LOG="logs/exp204_replace_watch.log"

echo "=========================================="
echo "启动 EXP-204 (DeePAW Replace 模式)"
echo "=========================================="
echo "配置:"
echo "  - DeePAW Fusion: replace"
echo "  - Learning Rate: 1e-4 (和 Baseline 一致)"
echo "  - n_rbf: 64 (和 Baseline 一致)"
echo "  - Batch Size: 64"
echo "  - Epochs: 50"
echo ""
echo "日志文件:"
echo "  - 训练日志: $TRAIN_LOG"
echo "  - 监控日志: $WATCH_LOG"
echo ""
echo "对比基准:"
echo "  - Baseline:     Val Loss 0.9781, Band Gap 0.2308 eV"
echo "  - EXP-201 (Add):    Val Loss 1.0739, Band Gap 0.2581 eV"
echo "  - EXP-202 (Concat): Val Loss 1.0430, Band Gap 0.2558 eV"
echo ""
echo "目标: Val Loss < 1.04, Band Gap < 0.25 eV"
echo "=========================================="
echo ""

# 启动训练
echo "启动训练进程..."
bash experiments/stage_a/phase2_deepaw/exp204_deepaw_replace/train.sh > "$TRAIN_LOG" 2>&1 &
TRAIN_PID=$!

echo "训练进程 PID: $TRAIN_PID"
echo ""

# 等待训练开始
echo "等待训练开始..."
sleep 10

# 启动监控
echo "启动监控进程..."
python scripts/watch_training_progress.py \
  --pattern "python scripts/train_multitask.py.*artifacts/runs_exp204" \
  --train-log "$TRAIN_LOG" \
  --watch-log "$WATCH_LOG" \
  --interval 30 \
  --also-append-to-train-log &
WATCH_PID=$!

echo "监控进程 PID: $WATCH_PID"
echo ""

echo "=========================================="
echo "训练已启动！"
echo "=========================================="
echo ""
echo "查看实时日志:"
echo "  tail -f $TRAIN_LOG"
echo ""
echo "查看训练进度:"
echo "  tail -f $WATCH_LOG"
echo ""
echo "或使用进度条:"
echo "  python scripts/show_progress_bar.py --watch-log $WATCH_LOG --interval 1 --width 30"
echo ""
echo "停止训练:"
echo "  kill $TRAIN_PID $WATCH_PID"
echo ""
echo "=========================================="
