#!/usr/bin/env bash
set -euo pipefail

FAMILY_DIR="experiments/stage_c/phase1_head_variants/family_stagec_head_variants"
STATUS_FILE="$FAMILY_DIR/WATCH_STATUS.md"
SNAPSHOT_LOG="$FAMILY_DIR/watch.log"
QUEUE_LOG="$FAMILY_DIR/queue.log"
CSV_FILE="$FAMILY_DIR/stagec_family_status.csv"
EXP_ROOT="experiments/stage_c/phase1_head_variants"

mkdir -p "$FAMILY_DIR"

exp_dirs=(
  "$EXP_ROOT/exp107_stagec_h1_electronic_hier"
  "$EXP_ROOT/exp108_stagec_h2_elastic_derived"
  "$EXP_ROOT/exp109_stagec_hybrid_hier_combo"
)

while true; do
  ts="$(date '+%Y-%m-%d %H:%M:%S')"
  queue_pid="$(pgrep -f '^python scripts/run_stagec_head_variants.py$' | head -n 1 || true)"
  current_train_pid="$(pgrep -f 'scripts/train_multitask.py.*--head-variant stagec_' | head -n 1 || true)"
  gpu_line="$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null | awk -F, -v pid="$current_train_pid" '$1 ~ pid {print $0}' | head -n 1 || true)"
  latest_queue_tail="$(tail -n 20 "$QUEUE_LOG" 2>/dev/null || true)"

  {
    echo "# Stage C Queue Watch Status"
    echo
    echo "- timestamp: $ts"
    echo "- queue_pid: ${queue_pid:-none}"
    echo "- current_train_pid: ${current_train_pid:-none}"
    echo "- gpu: ${gpu_line:-none}"
    echo "- family_csv: \
\
\`$CSV_FILE\`"
    echo
    echo "## Experiments"
    echo
    echo "| Experiment | Head Variant | Status | Latest Epoch | Latest Val Loss | Run Dir |"
    echo "|---|---|---|---:|---:|---|"

    all_done=true
    for exp_dir in "${exp_dirs[@]}"; do
      exp_name="$(basename "$exp_dir")"
      status_md="$exp_dir/TRAINING_STATUS.md"
      train_log="$exp_dir/logs/train.log"
      head_variant="$(grep -E '^- head_variant:' "$status_md" 2>/dev/null | sed -E 's/^- head_variant: `([^`]*)`/\1/' || true)"
      best_run="$(grep -E '^- run_dir:' "$status_md" 2>/dev/null | sed -E 's/^- run_dir: `([^`]*)`/\1/' || true)"
      [[ -z "$best_run" ]] && best_run="N/A"
      status="planned"
      latest_epoch="N/A"
      latest_val="N/A"
      if [[ -f "$train_log" ]]; then
        epoch_line="$(grep '^epoch=' "$train_log" | tail -n 1 || true)"
        if [[ -n "$epoch_line" ]]; then
          latest_epoch="$(printf '%s' "$epoch_line" | sed -E 's/^epoch=0*([0-9]+).*/\1/')"
          latest_val="$(printf '%s' "$epoch_line" | sed -E 's/.*val_loss=([0-9.]+).*/\1/')"
          status="running"
        fi
        if grep -q 'run_dir=' "$train_log" 2>/dev/null; then
          status="completed"
        fi
        if grep -Eq 'Traceback|ValueError: NaN/Inf|RuntimeError' "$train_log" 2>/dev/null; then
          status="failed"
        fi
      fi
      if [[ "$status" != "completed" ]]; then
        all_done=false
      fi
      echo "| $exp_name | ${head_variant:-unknown} | $status | $latest_epoch | $latest_val | $best_run |"
    done

    echo
    echo "## Queue Tail"
    echo '```text'
    printf '%s\n' "$latest_queue_tail"
    echo '```'
  } > "$STATUS_FILE"

  echo "[$ts] queue_pid=${queue_pid:-none} train_pid=${current_train_pid:-none} gpu=${gpu_line:-none}" >> "$SNAPSHOT_LOG"

  if [[ -z "${queue_pid:-}" && "$all_done" == true ]]; then
    exit 0
  fi

  sleep 60
done
