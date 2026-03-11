# Experiment 105: Stage A PyG Baseline

**Date**: 2026-03-08
**Status**: Ready
**Phase**: 3 (Stage A PyG)

## Objective
Retrain the Stage A baseline with the same model, optimization settings, and exact 8-task scope as `20260305_210307`, but using the PyG backend under fully isolated outputs.

## Isolation Rules
- Dedicated config: `configs/exp105_stagea_pyg_baseline.json`
- Dedicated experiment root: `experiments/stage_a/phase3_pyg/exp105_stagea_pyg_baseline/`
- Dedicated run root: `artifacts/runs_stagea_pyg/`
- Dedicated logs and reports
- No overwrite of prior Stage A baseline artifacts

## Reference Run
- Non-PyG baseline: `artifacts/runs/20260305_210307/`

## Runtime
- Training PID: `487374`
- Training log: `experiments/stage_a/phase3_pyg/exp105_stagea_pyg_baseline/logs/train.log`

## Exact Task Parity
- Exclude tasks: `volume`, `density`, `is_stable`
- Enabled tasks match `artifacts/runs/20260305_210307/config.json` exactly
