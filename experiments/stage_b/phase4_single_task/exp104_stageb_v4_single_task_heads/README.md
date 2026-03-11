# Experiment 104: Stage B v4 Single-Task Heads

**Date**: 2026-03-08
**Status**: Completed
**Phase**: 4 (Stage B v4)

## Objective
Use a shared pretrained backbone checkpoint and run isolated single-task fine-tuning jobs (one task per run) to reduce cross-task interference.

## Isolation Rules
- Dedicated config: `configs/exp104_stageb_v4_single_task_heads.json`
- Dedicated experiment root: `experiments/stage_b/phase4_single_task/exp104_stageb_v4_single_task_heads/`
- Dedicated run root: `artifacts/runs_stageb_v4/<task>/<run_id>/`
- Per-task dedicated logs and command files
- No overwrite of Stage B v1/v2/v3 artifacts

## Base Checkpoint
- `artifacts/runs_stageb_v3/20260308_070539/checkpoints/best.pt`

## Runner
- Script: `scripts/run_stageb_v4_single_task.py`
- Status file: `TRAINING_STATUS.md`
- Metrics registry: `metrics/single_task_runs.csv`
- Manifest: `manifest.json`

## Start Command
See `training_cmd.sh`.

## Runtime
- Orchestrator PID: `completed`
- Orchestrator log: `experiments/stage_b/phase4_single_task/exp104_stageb_v4_single_task_heads/orchestrator.log`
- Per-task logs: `experiments/stage_b/phase4_single_task/exp104_stageb_v4_single_task_heads/logs/<task>.log`

## Outputs
- Unified comparison report: `reports/V4_UNIFIED_COMPARISON_STAGEA_V1_V2_V3_V4.md`
- Unified comparison CSV: `reports/v4_unified_comparison_stageA_v1_v2_v3_v4.csv`
