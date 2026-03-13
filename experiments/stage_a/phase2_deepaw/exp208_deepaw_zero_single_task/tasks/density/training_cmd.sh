#!/bin/bash
# EXP-208: DeePAW Zero Single-Task - density
# Capacity: 320×7, Samples: 123902

set -e
export PYTHONNOUSERSITE=1

python scripts/train_multitask.py --db /scratch/sutianhao/data/mp-data-pipeline/data/db/mp_materials.db --split /scratch/sutianhao/data/mp-data-pipeline/data/splits/split_iid_seed42.json --stage full --only-task density --backbone enhanced_graph --use-deepaw-features --deepaw-checkpoint /home/sutianhao/data/deepaw_test/DeePAW-main/checkpoints/f_nonlocal_escn_best.pth --deepaw-fusion replace --hidden-dim 320 --layers 7 --cutoff 6.0 --max-neighbors 24 --n-rbf 128 --batch-size 32 --epochs 70 --lr 8e-05 --weight-decay 1e-05 --num-workers 4 --warmup-epochs 5 --grad-clip 0.5 --device cuda --out-dir /scratch/sutianhao/data/mp-data-pipeline/artifacts/runs_exp208/density --use-pyg
