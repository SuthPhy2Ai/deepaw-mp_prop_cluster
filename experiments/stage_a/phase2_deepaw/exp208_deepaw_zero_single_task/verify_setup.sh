#!/bin/bash
# EXP-208 Verification: Test band_gap task with 1 epoch

set -e
export PYTHONNOUSERSITE=1

echo "========================================="
echo "EXP-208 Verification Test"
echo "========================================="
echo "Task: band_gap"
echo "Epochs: 1 (quick test)"
echo "Purpose: Verify DeePAW integration works"
echo "========================================="
echo ""

python scripts/train_multitask.py \
  --db /scratch/sutianhao/data/mp-data-pipeline/data/db/mp_materials.db \
  --split /scratch/sutianhao/data/mp-data-pipeline/data/splits/split_iid_seed42.json \
  --stage full \
  --only-task band_gap \
  --backbone enhanced_graph \
  --use-deepaw-features \
  --deepaw-checkpoint /home/sutianhao/data/deepaw_test/DeePAW-main/checkpoints/f_nonlocal_escn_best.pth \
  --deepaw-fusion replace \
  --hidden-dim 256 \
  --layers 6 \
  --cutoff 6.0 \
  --max-neighbors 24 \
  --n-rbf 128 \
  --batch-size 64 \
  --epochs 1 \
  --lr 0.0001 \
  --weight-decay 1e-05 \
  --num-workers 4 \
  --warmup-epochs 0 \
  --grad-clip 1.0 \
  --device cuda \
  --out-dir /tmp/exp208_verification_test \
  --use-pyg

echo ""
echo "========================================="
echo "✅ Verification test completed!"
echo "========================================="
echo "If you see this message, exp208 is ready to run."
echo "To start full training:"
echo "  bash experiments/stage_a/phase2_deepaw/exp208_deepaw_zero_single_task/training_cmd.sh"
echo "========================================="
