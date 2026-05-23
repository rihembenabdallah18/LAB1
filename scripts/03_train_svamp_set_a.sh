#!/usr/bin/env bash
# Stage 3 (SVAMP): fine-tune FLAN-T5-base on SVAMP Set A (no filter).
set -euo pipefail
cd "$(dirname "$0")/.."
python -m src.train.finetune \
  --config config/config.yaml \
  --train data/processed/svamp/svamp_set_a_nofilter.jsonl \
  --run-name svamp_student_set_a \
  "$@"
