#!/usr/bin/env bash
# Stage 3 (SVAMP): fine-tune FLAN-T5-base on SVAMP Direct FT (answer-only).
set -euo pipefail
cd "$(dirname "$0")/.."
python -m src.train.finetune \
  --config config/config.yaml \
  --train data/processed/svamp/svamp_direct_ft.jsonl \
  --run-name svamp_student_direct_ft \
  "$@"
