#!/bin/bash
# One SLURM task for scripts/rank_272_landing_targets.py (notebook 272, Question 6):
# rank the target of every report-half kmerseek landing call among everything that
# search hit, reading the midi-plus region tables in place.
#
#   cd $SCRATCH/244-hero && sbatch code/scripts/rank_272_landing_targets.sbatch
#SBATCH --job-name=272-ranks
#SBATCH --account=ayeletv
#SBATCH --partition=hns
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.log

set -euo pipefail
RESULTS=/scratch/users/olgabot/2024-kmerseek-analysis/nextflow-runs/qfo-pfam-region-benchmark/data/midi-plus/results
IMAGE=/scratch/users/olgabot/apptainer-cache/docker.io-olgabot-kmerseek-0.4.0-rc5.img
export POLARS_MAX_THREADS=${SLURM_CPUS_PER_TASK}

apptainer exec --bind /scratch "$IMAGE" python3 code/scripts/rank_272_landing_targets.py \
    --results "$RESULTS" \
    --want code/tables/272_landing_targets_to_rank.csv \
    --out 272_landing_target_ranks.csv
