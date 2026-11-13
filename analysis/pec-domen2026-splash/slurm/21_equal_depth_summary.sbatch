#!/bin/bash
#SBATCH --job-name=nb361-ed-summary
#SBATCH --account=ayeletv
#SBATCH --output=%x.%j.out
# Counts, reached genes and control recall for the equal-depth test (scripts/equal_depth_summary.py).
set -euo pipefail
: "${PEC_ROOT:?}" "${CODE_DIR:?}"
"$PEC_ROOT/env/bin/python" "$CODE_DIR/scripts/equal_depth_summary.py" "$PEC_ROOT" \
  "$PEC_ROOT/db/botznik.id_map.tsv" "$PEC_ROOT/equal_depth/summary" > "$PEC_ROOT/equal_depth/summary.log" 2>&1
