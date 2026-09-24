#!/bin/bash
# SLURM array for scripts/reduce_disprot_region_landing.py (notebook 251).
#
# Reads every kmerseek arm and the seven comparison tools of `make run-disprot` and writes
# four small parquets per (arm, target) under landing_disprot/. The work directory holds
# code/ (from git archive), disprot/ (the tables scripts/fetch_disprot.py wrote on the
# Mac: disprot_function_regions.parquet and disprot_proteins.parquet) and logs/:
#
#   cd $SCRATCH/251-disprot && sbatch code/scripts/reduce_disprot_region_landing.sbatch
#
# A task skips any (arm, target) whose outputs already exist, so resubmitting the same
# array after a failure only redoes what is missing.
#SBATCH --job-name=251-landing
#SBATCH --account=ayeletv
#SBATCH --partition=hns
#SBATCH --array=0-99
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%A_%a.log

set -euo pipefail
RESULTS=/scratch/users/olgabot/2024-kmerseek-analysis/nextflow-runs/qfo-pfam-region-benchmark/data/disprot/results
IMAGE=/scratch/users/olgabot/apptainer-cache/docker.io-olgabot-kmerseek-0.4.0-rc5.img
export POLARS_MAX_THREADS=${SLURM_CPUS_PER_TASK}

apptainer exec --bind /scratch "$IMAGE" python3 code/scripts/reduce_disprot_region_landing.py \
    --results "$RESULTS" \
    --disprot-dir disprot \
    --qfo-bin code/nextflow-runs/qfo-pfam-region-benchmark/bin \
    --out-dir landing_disprot \
    --task "${SLURM_ARRAY_TASK_ID}" \
    --n-tasks "${SLURM_ARRAY_TASK_COUNT}"
