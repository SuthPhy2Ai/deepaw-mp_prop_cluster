#!/usr/bin/env bash
set -euo pipefail

ROOT="/scratch/sutianhao/data/mp-data-pipeline"
cd "${ROOT}"

ts() { date +"%Y%m%d_%H%M%S"; }

rotate_log() {
  local f="$1"
  if [ -f "${f}" ]; then
    mv "${f}" "${f%.log}.bak_$(ts).log"
  fi
}

run_one() {
  local exp_name="$1"   # exp202 / exp203
  local train_sh="$2"
  local out_dir="$3"    # artifacts/runs_exp202
  local train_log="$4"
  local watch_log="$5"

  rotate_log "${train_log}"
  rotate_log "${watch_log}"

  # Start watcher first (it will show stopped until python starts).
  nohup python scripts/watch_training_progress.py \
    --pattern "python scripts/train_multitask.py.*${out_dir}" \
    --train-log "${train_log}" \
    --watch-log "${watch_log}" \
    --interval 30 \
    --also-append-to-train-log \
    > "${watch_log%.log}_stdout.log" 2>&1 &
  local watch_pid=$!
  echo "${watch_pid}" > "logs/${exp_name}_watch.pid"

  echo "== $(date) starting ${exp_name} ==" >> "${train_log}"
  bash "${train_sh}" >> "${train_log}" 2>&1
  echo "== $(date) finished ${exp_name} ==" >> "${train_log}"

  kill "${watch_pid}" 2>/dev/null || true
}

mkdir -p logs

run_one \
  "exp202" \
  "experiments/stage_a/phase2_deepaw/exp202_deepaw_concat/train.sh" \
  "artifacts/runs_exp202" \
  "logs/exp202_concat_c6n24.log" \
  "logs/exp202_watch_c6n24.log"

run_one \
  "exp203" \
  "experiments/stage_a/phase2_deepaw/exp203_deepaw_angles/train.sh" \
  "artifacts/runs_exp203" \
  "logs/exp203_angles_c6n24.log" \
  "logs/exp203_watch_c6n24.log"

echo "All done."

