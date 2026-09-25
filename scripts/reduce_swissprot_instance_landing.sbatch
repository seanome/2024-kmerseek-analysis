#!/bin/bash
# SLURM array for scripts/reduce_swissprot_instance_landing.py (notebook 244, Stage 0).
#
# Reads every kmerseek arm (mask on and off) and six comparison tools for the nine midi-plus
# species, and writes one small per-instance parquet per (arm, species) under landing/.
# Submit from the work directory, which holds code/ (from git archive) and logs/:
#
#   cd $SCRATCH/244-hero && sbatch code/scripts/reduce_swissprot_instance_landing.sbatch
#
# A task skips any (arm, species) whose outputs already exist, so resubmitting the same
# array after a failure only redoes what is missing.
#SBATCH --job-name=244-landing
#SBATCH --account=ayeletv
#SBATCH --partition=hns
#SBATCH --array=0-99
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%A_%a.log

set -euo pipefail
RESULTS=/scratch/users/olgabot/2024-kmerseek-analysis/nextflow-runs/qfo-pfam-region-benchmark/data/midi-plus/results
IMAGE=/scratch/users/olgabot/apptainer-cache/docker.io-olgabot-kmerseek-0.4.0-rc5.img
export POLARS_MAX_THREADS=${SLURM_CPUS_PER_TASK}

apptainer exec --bind /scratch "$IMAGE" python3 code/scripts/reduce_swissprot_instance_landing.py \
    --results "$RESULTS" \
    --qfo-bin code/nextflow-runs/qfo-pfam-region-benchmark/bin \
    --out-dir landing \
    --task "${SLURM_ARRAY_TASK_ID}" \
    --n-tasks "${SLURM_ARRAY_TASK_COUNT}"
