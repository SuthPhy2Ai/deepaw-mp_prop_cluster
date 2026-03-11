# Experiment 102: Stage B v2 Balanced

**Date**: 2026-03-08
**Status**: Completed
**Phase**: 3 (Stage B v2)

## Objective
Evaluate whether reducing elastic oversampling (4.0 -> 2.0) can recover Stage A core task performance while preserving elastic-task learnability.

## Isolation Rules
- Dedicated config: `configs/exp102_stageb_v2_balanced.json`
- Dedicated run output root: `artifacts/runs_stageb_v2/`
- Dedicated logs and analysis under this experiment directory
- No overwrite of previous Stage B artifacts

## Training Command
See `training_cmd.sh` in this folder.

## Comparison Targets
- Stage A baseline: run `20260305_210307`
- Stage B v1 baseline: run `20260307_185342`

## Expected Deliverables
- best checkpoint + history + best_summary
- full evaluation (train/val/test)
- side-by-side numeric comparison report

## Execution Result
- Run ID: `20260308_001437`
- Output root: `artifacts/runs_stageb_v2/20260308_001437`
- Best epoch: `42`
- Best val loss: `26.9097`

## Final Artifacts
- Training log: `training_log.txt`
- Evaluation outputs: `analysis/`
- Comparison report: `metrics/COMPARISON_V2_REPORT.md`
- Comparison csv: `metrics/comparison_v2_vs_v1_stagea.csv`
