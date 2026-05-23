#!/usr/bin/env bash
# Stage 4 (SVAMP): inference on the SVAMP test set for all conditions.
#
# Two groups of conditions are evaluated:
#   (a) GSM8K-trained models tested on SVAMP (cross-dataset transfer)
#   (b) SVAMP-trained models tested on SVAMP (in-distribution)
#
# Each condition is resumable - safe to re-run after a session timeout.
set -euo pipefail
cd "$(dirname "$0")/.."

# (a) GSM8K-trained models → SVAMP test set (zero-shot transfer)
GSM8K_TRAINED=(
  baseline
  student_direct_ft
  student_set_a
  student_set_b
  student_set_c
)

# (b) SVAMP-trained models → SVAMP test set (in-distribution)
SVAMP_TRAINED=(
  svamp_student_direct_ft
  svamp_student_set_a
  svamp_student_set_b
  svamp_student_set_c
)

echo "=== GSM8K-trained → SVAMP test set ==="
for COND in "${GSM8K_TRAINED[@]}"; do
  echo "--- ${COND} ---"
  python -m src.inference.generate --dataset svamp --condition "${COND}"
done

echo "=== SVAMP-trained → SVAMP test set ==="
for COND in "${SVAMP_TRAINED[@]}"; do
  echo "--- ${COND} ---"
  python -m src.inference.generate --dataset svamp --condition "${COND}"
done

echo "=== all done ==="
ls -lh outputs/generations/svamp/
python -m src.status
