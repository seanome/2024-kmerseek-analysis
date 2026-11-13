#!/bin/bash
#SBATCH --job-name=nb361-share-pair
#SBATCH --partition=hns,normal
#SBATCH --account=ayeletv
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=%x.%j.out
# Per-sample counts of each kept SPLASH anchor's two top targets (from SPLASH's satc dump): does the
# second target's share follow the colony pair or the treatment? Needs call_types.py output first.
set -euo pipefail
: "${PEC_ROOT:?}" "${PEC_CODE:?}"
export PATH=$PEC_ROOT/env/bin:$PATH
export POLARS_MAX_THREADS=$SLURM_CPUS_PER_TASK
python "$PEC_CODE/scripts/target_share_by_pair.py" "$PEC_ROOT/contrasts" "$PEC_ROOT/splash" \
  "$PEC_ROOT/call_types/target_distances.parquet" "$PEC_ROOT/call_types"
