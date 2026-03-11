python scripts/run_stageb_v4_single_task.py \
  --base-checkpoint artifacts/runs_stageb_v3/20260308_070539/checkpoints/best.pt \
  --experiment-dir experiments/stage_b/phase4_single_task/exp104_stageb_v4_single_task_heads \
  --runs-root artifacts/runs_stageb_v4 \
  --epochs 30 \
  --batch-size 64 \
  --num-workers 4 \
  --device cuda
