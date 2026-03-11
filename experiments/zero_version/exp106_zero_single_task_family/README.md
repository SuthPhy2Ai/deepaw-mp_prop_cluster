# Experiment 106: Zero Single-Task Family

**Date**: 2026-03-08
**Status**: Ready
**Phase**: Zero Version

## Objective
Train one fully independent model per property with no shared backbone. Model capacity is sized per task according to sample count and task characteristics.

## Scope
- Full task scope: all `15` tasks in `TASK_NAME_LIST`
- No shared initialization
- No shared backbone
- Dedicated run root per task

## Capacity Policy
- `volume`, `density`: `hidden_dim=320`, `layers=7`
- `train_samples >= 100k`: `hidden_dim=256`, `layers=6`
- `train_samples >= 50k`: `hidden_dim=224`, `layers=6`
- otherwise: `hidden_dim=160`, `layers=4`

## Isolation
- Experiment root: `experiments/zero_version/exp106_zero_single_task_family/`
- Run root: `artifacts/runs_zero/<task>/<run_id>/`
- Metrics registry: `metrics/zero_runs.csv`
