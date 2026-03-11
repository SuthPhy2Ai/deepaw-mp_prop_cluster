# Zero Retry Recovery Log

- parent_experiment: `experiments/zero_version/exp106_zero_single_task_family`
- retry_experiment: `experiments/zero_version/exp106_zero_single_task_family_retry_sparse`
- recovered_tasks:
  - `homogeneous_poisson`
  - `universal_anisotropy`

## Root Cause

The original zero-policy configuration for sparse tasks used `hidden_dim=160`, `layers=4`,
`lr=2e-4`, and AMP enabled. Both recovered tasks hit `NaN/Inf` during mid-training, which
points to numerical instability rather than missing labels or data loading failures.

## Retry Policy

- `hidden_dim=128`
- `layers=4`
- `batch_size=32`
- `epochs=90`
- `lr=8e-5`
- `grad_clip=0.5`
- `--no-amp`

## Recovered Results

- `homogeneous_poisson`
  - run_dir: `artifacts/runs_zero_retry_sparse/homogeneous_poisson/20260309_132811`
  - best_epoch: `22`
  - best_val_loss: `0.0015998720114029864`
  - val_primary_metric: `0.049809928983449936`
- `universal_anisotropy`
  - run_dir: `artifacts/runs_zero_retry_sparse/universal_anisotropy/20260309_140536`
  - best_epoch: `30`
  - best_val_loss: `0.7314344132238295`
  - val_primary_metric: `1.8494423627853394`

The master registry in `metrics/zero_runs.csv` has been updated to reflect these recovered runs
while preserving the isolated retry experiment directory for auditability.
