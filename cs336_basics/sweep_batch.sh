#!/usr/bin/env bash
# Batch-size sweep at fixed lr=0.001 (static LR, no decay).
# Fair comparison: every run trains for the SAME wall-clock time
# (MAX_TRAIN_SECONDS) regardless of how many steps it fits in.
# Each invocation is a separate wandb run, named lr=0.001-bs=<value>,
# all sharing one wandb group so they cluster in the UI.
set -euo pipefail

PYTHON="$HOME/tf_venv/bin/python"
GROUP="batch-sweep-$(date +%Y%m%d-%H%M%S)"
LR="0.001"
MAX_TRAIN_SECONDS="${MAX_TRAIN_SECONDS:-7200}"  # 2 hr (7200s) per run; override via env

batch_sizes=(64 150)

for bs in "${batch_sizes[@]}"; do
  echo "=== sweep: batch_size=$bs lr=$LR time=${MAX_TRAIN_SECONDS}s group=$GROUP ==="
  "$PYTHON" training_together.py --lr "$LR" --batch_size "$bs" \
      --max_train_seconds "$MAX_TRAIN_SECONDS" --group "$GROUP" "$@"
done
