# Experiment 103: Stage B v3 Core Guard

**Date**: 2026-03-08
**Status**: Completed
**Phase**: 3 (Stage B v3)

## Objective
Test Stage B v3 with `oversample_elastic=1.0` to further prioritize Stage A core-task retention.

## Isolation Rules
- Dedicated config: `configs/exp103_stageb_v3_core_guard.json`
- Dedicated run output root: `artifacts/runs_stageb_v3/`
- Dedicated logs, metrics, analysis in this experiment folder
- No overwrite of v1/v2 artifacts

## Training Command
See `training_cmd.sh`.

## Final Run
- Run ID: `20260308_070539`
- Run root: `artifacts/runs_stageb_v3/20260308_070539/`
- Best checkpoint: `artifacts/runs_stageb_v3/20260308_070539/checkpoints/best.pt`
- Best summary: `artifacts/runs_stageb_v3/20260308_070539/metrics/best_summary.json`

## Comparison Targets
- Stage A baseline: `20260305_210307`
- Stage B v1 baseline: `20260307_185342`
- Stage B v2: `20260308_001437`

## Deliverables
- Training completion mark: `TRAINING_COMPLETE.md`
- Training log: `training_log.txt`
- Isolated comparison report: `metrics/COMPARISON_V3_REPORT.md`
- Isolated comparison CSV: `metrics/comparison_v3_vs_v2_v1_stagea.csv`
- Full evaluation outputs (Train/Val/Test):
  - `reports/gpt_eval_20260308_070539/results.json`
  - `reports/gpt_eval_20260308_070539/analysis_report.md`
  - `reports/gpt_eval_20260308_070539/performance_visualization.png`
- GPT evaluation package:
  - `reports/GPT_EVAL_20260308_070539.md`
  - `reports/gpt_eval_20260308_070539_vs_20260305_210307_metrics.csv`
  - `reports/figures/gpt_eval_20260308_070539_scatter_val_test.png`
  - `reports/figures/gpt_eval_20260308_070539_roc_val_test.png`
  - `reports/figures/gpt_eval_20260308_070539_vs_20260305_210307_table.png`
