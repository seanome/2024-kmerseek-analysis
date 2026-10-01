#!/bin/bash
# One SLURM task for scripts/extract_244_case_hits.py: cut the 33 midi-plus kmerseek region
# tables (21 GB) that the notebook-244 cases' chosen arms need down to the rows for the
# cases' human proteins, so the cut can be copied to the laptop for the hits pages.
#
#   cd $SCRATCH/244-hero && sbatch code/scripts/extract_244_case_hits.sbatch
#
# Tables already cut are skipped, so resubmitting after a failure only redoes what is left.
#SBATCH --job-name=244-case-hits
#SBATCH --account=ayeletv
#SBATCH --partition=hns
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.log

set -euo pipefail
RESULTS=/scratch/users/olgabot/2024-kmerseek-analysis/nextflow-runs/qfo-pfam-region-benchmark/data/midi-plus/results
IMAGE=/scratch/users/olgabot/apptainer-cache/docker.io-olgabot-kmerseek-0.4.0-rc5.img
export POLARS_MAX_THREADS=${SLURM_CPUS_PER_TASK}

apptainer exec --bind /scratch "$IMAGE" python3 code/scripts/extract_244_case_hits.py \
    --results "$RESULTS" \
    --candidates code/tables/244_hero_candidates.csv \
    --out-dir case_hits
