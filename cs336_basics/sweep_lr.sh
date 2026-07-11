#!/usr/bin/env bash
# Static learning-rate sweep (no warmup, no decay).
# Each entry is a single lr. Each invocation is a separate wandb run,
# named lr=<value>, all sharing one wandb group so they cluster in the UI.
set -euo pipefail

PYTHON="$HOME/tf_venv/bin/python"
GROUP="static-lr-sweep-$(date +%Y%m%d-%H%M%S)"

runs=(
  "1e-5"
  "3e-5"
  "1e-4"
  "3e-4"
  "1e-3"
  "3e-3"
  "1e-2"
  "3e-2"
  "1e-1"
  "3e-1"
  "1"
  "3"
  "10"
)

for lr in "${runs[@]}"; do
  echo "=== sweep: lr=$lr group=$GROUP ==="
  "$PYTHON" training_together.py --lr "$lr" --group "$GROUP" "$@"
done
