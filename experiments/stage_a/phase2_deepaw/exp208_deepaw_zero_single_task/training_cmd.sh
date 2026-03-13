#!/bin/bash
# EXP-208: DeePAW Zero Single-Task Family - Master Training Script
# This script orchestrates training for all 15 tasks

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

echo "========================================="
echo "EXP-208: DeePAW Zero Single-Task Family"
echo "========================================="
echo "Experiment: Enhanced Graph + DeePAW Replace mode"
echo "Tasks: 15 independent models"
echo "Capacity: Task-specific (160-320 dim)"
echo "========================================="
echo ""

# Check if running in dry-run mode
DRY_RUN=""
if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN="--dry-run"
    echo "🔍 DRY-RUN MODE: Will generate scripts only, no training"
    echo ""
fi

# Run orchestration script
cd "$PROJECT_ROOT"
python scripts/run_exp208_deepaw_zero.py $DRY_RUN

echo ""
echo "========================================="
echo "✅ EXP-208 Complete!"
echo "========================================="
echo "Results: experiments/stage_a/phase2_deepaw/exp208_deepaw_zero_single_task/metrics/exp208_results.csv"
echo "Logs: experiments/stage_a/phase2_deepaw/exp208_deepaw_zero_single_task/logs/"
echo "Checkpoints: artifacts/runs_exp208/<task>/"
echo "========================================="
