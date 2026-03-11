#!/bin/bash
# 批量运行所有 DeePAW 实验
# 按优先级顺序执行

set -e

echo "========================================="
echo "DeePAW Training Experiments - Batch Run"
echo "========================================="
echo ""
echo "Total experiments: 6"
echo "Estimated total time: ~9 hours"
echo ""
echo "========================================="

# 第一优先级：必须执行
echo ""
echo "=== Priority 1: Core Experiments ==="
echo ""

echo "[1/6] Running EXP-201: DeePAW Add Fusion (50 epochs)..."
bash experiments/stage_a/phase2_deepaw/exp201_deepaw_add/train.sh
echo "✓ EXP-201 completed"
echo ""

echo "[2/6] Running EXP-202: DeePAW Concat Fusion (50 epochs)..."
bash experiments/stage_a/phase2_deepaw/exp202_deepaw_concat/train.sh
echo "✓ EXP-202 completed"
echo ""

# 第二优先级：推荐执行
echo ""
echo "=== Priority 2: Extended Experiments ==="
echo ""

echo "[3/6] Running EXP-203: DeePAW + Angles (50 epochs)..."
bash experiments/stage_a/phase2_deepaw/exp203_deepaw_angles/train.sh
echo "✓ EXP-203 completed"
echo ""

echo "[4/6] Running EXP-206: DeePAW Stage B (100 epochs)..."
bash experiments/stage_b/phase2_deepaw/exp206_deepaw_stageb/train.sh
echo "✓ EXP-206 completed"
echo ""

# 第三优先级：可选执行
echo ""
echo "=== Priority 3: Optional Experiments ==="
echo ""

echo "[5/6] Running EXP-204: DeePAW Long Training (100 epochs)..."
bash experiments/stage_a/phase2_deepaw/exp204_deepaw_long/train.sh
echo "✓ EXP-204 completed"
echo ""

echo "[6/6] Running EXP-205: DeePAW Lower LR (50 epochs)..."
bash experiments/stage_a/phase2_deepaw/exp205_deepaw_lr1e4/train.sh
echo "✓ EXP-205 completed"
echo ""

echo "========================================="
echo "All experiments completed successfully!"
echo "========================================="
echo ""
echo "Results saved to:"
echo "  - artifacts/runs_exp201/"
echo "  - artifacts/runs_exp202/"
echo "  - artifacts/runs_exp203/"
echo "  - artifacts/runs_exp204/"
echo "  - artifacts/runs_exp205/"
echo "  - artifacts/runs_exp206/"
echo ""
echo "Run comparison script:"
echo "  python experiments/stage_a/phase2_deepaw/compare_results.py"
echo ""
