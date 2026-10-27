#!/bin/bash
#SBATCH --job-name=nb361-download
#SBATCH --partition=hns,normal
#SBATCH --account=ayeletv
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=08:00:00
#SBATCH --output=%x.%A_%a.out
# One array task per ENA run (line N+1 of meta/runs.txt). Writes reads/<run>.done only after
# MD5 and read counts agree with ENA. Restartable: existing reads are reverified before markers are reused.
set -euo pipefail
: "${PEC_ROOT:?}" "${PEC_CODE:?}"
cd "$PEC_ROOT"
mkdir -p logs
RUN=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" meta/runs.txt)
[ -n "$RUN" ] || { echo "no run for task $SLURM_ARRAY_TASK_ID"; exit 1; }
export PATH=$PEC_ROOT/env/bin:$PATH
/usr/bin/time -v python "$PEC_CODE/scripts/download_run.py" meta/ena_filereport.tsv "$RUN" reads \
  "${PEC_REUSE_DIR:-}" 2> "logs/download_$RUN.time"
