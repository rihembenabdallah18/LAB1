#!/usr/bin/env bash
# Stage 3 (SVAMP): fine-tune FLAN-T5-base on SVAMP Set B (Magister filter).
set -euo pipefail
cd "$(dirname "$0")/.."
python -m src.train.finetune \
  --config config/config.yaml \
  --train data/processed/svamp/svamp_set_b_magister.jsonl \
  --run-name svamp_student_set_b \
  "$@"
