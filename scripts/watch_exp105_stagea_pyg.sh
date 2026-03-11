#!/usr/bin/env bash
set -euo pipefail

EXP_DIR="experiments/stage_a/phase3_pyg/exp105_stagea_pyg_baseline"
LOG_FILE="$EXP_DIR/logs/train.log"
STATUS_FILE="$EXP_DIR/WATCH_STATUS.md"
SNAPSHOT_LOG="$EXP_DIR/logs/watch.log"
RUNS_ROOT="artifacts/runs_stagea_pyg"
TRAIN_PATTERN="scripts/train_multitask.py --db /scratch/sutianhao/data/mp-data-pipeline/data/db/mp_materials.db --split data/splits/split_iid_seed42.json --stage a"

mkdir -p "$EXP_DIR/logs"

while true; do
  ts="$(date '+%Y-%m-%d %H:%M:%S')"
  train_pid="$(pgrep -f "$TRAIN_PATTERN" | head -n 1 || true)"
  latest_run="$(find "$RUNS_ROOT" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | sort | tail -n 1 || true)"
  gpu_line="$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null | rg "^${train_pid}," || true)"
  last_lines="$(tail -n 12 "$LOG_FILE" 2>/dev/null || true)"

  {
    echo "# Exp105 Watch Status"
    echo
    echo "- timestamp: $ts"
    echo "- train_pid: ${train_pid:-none}"
    echo "- latest_run: ${latest_run:-none}"
    echo "- gpu: ${gpu_line:-none}"
    echo
    echo "## Log Tail"
    echo '```text'
    printf '%s\n' "$last_lines"
    echo '```'
  } > "$STATUS_FILE"

  {
    echo "[$ts] pid=${train_pid:-none} run=${latest_run:-none} gpu=${gpu_line:-none}"
  } >> "$SNAPSHOT_LOG"

  if [[ -z "${train_pid:-}" ]]; then
    if [[ -n "${latest_run:-}" && -f "$latest_run/metrics/best_summary.json" ]]; then
      {
        echo
        echo "## Completion"
        echo "- best_summary: \`${latest_run}/metrics/best_summary.json\`"
      } >> "$STATUS_FILE"
    fi
    exit 0
  fi

  sleep 60
done
