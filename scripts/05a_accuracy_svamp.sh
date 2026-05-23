#!/usr/bin/env bash
# Stage 5a (SVAMP): accuracy scoring on SVAMP generations.
# Reads outputs/generations/svamp/<condition>.jsonl and writes
#   outputs/eval_results/svamp/svamp_accuracy.csv
#   outputs/plots/svamp_accuracy_bar.png
# Pure-CPU; takes seconds. Safe to re-run.
set -euo pipefail
cd "$(dirname "$0")/.."

python -m src.eval.accuracy --dataset svamp "$@"

echo
echo "=== outputs/eval_results/svamp/svamp_accuracy.csv ==="
cat outputs/eval_results/svamp/svamp_accuracy.csv
echo
python -m src.status
