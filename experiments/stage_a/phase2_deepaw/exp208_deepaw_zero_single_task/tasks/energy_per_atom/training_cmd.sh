#!/bin/bash
# EXP-208: DeePAW Zero Single-Task - energy_per_atom
# Capacity: 256×6, Samples: 123902

set -e
export PYTHONNOUSERSITE=1

python scripts/train_multitask.py --db /scratch/sutianhao/data/mp-data-pipeline/data/db/mp_materials.db --split /scratch/sutianhao/data/mp-data-pipeline/data/splits/split_iid_seed42.json --stage full --only-task energy_per_atom --backbone enhanced_graph --use-deepaw-features --deepaw-checkpoint /home/sutianhao/data/deepaw_test/DeePAW-main/checkpoints/f_nonlocal_escn_best.pth --deepaw-fusion replace --hidden-dim 256 --layers 6 --cutoff 6.0 --max-neighbors 24 --n-rbf 128 --batch-size 64 --epochs 50 --lr 0.0001 --weight-decay 1e-05 --num-workers 4 --warmup-epochs 5 --grad-clip 1.0 --device cuda --out-dir /scratch/sutianhao/data/mp-data-pipeline/artifacts/runs_exp208/energy_per_atom --use-pyg
