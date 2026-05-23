#!/usr/bin/env bash
# Stage 2 (SVAMP): build Set A / B / C / Direct-FT from SVAMP train + Ho et al. CoTs.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m src.data.filter --dataset svamp "$@"
