#!/bin/bash
# SLURM array for scripts/reduce_elm_landing.py (notebook 250).
#
# Scores every kmerseek arm (alphabet x k x mask on/off x scaled 1/2/5/10), the eight
# comparison tools and the Kyte-Doolittle scan from the ELM run, and writes three small
# parquets per (arm, target set) under landing/. Submitted by `make reduce-elm-motif` in
# nextflow-runs/qfo-pfam-region-benchmark, from a work directory holding code/ (a git
# archive of HEAD) and logs/.
#
# A task skips any (arm, target set) whose instances file already exists, so resubmitting
# the same array after a failure only redoes what is missing.
#SBATCH --job-name=250-elm-landing
#SBATCH --account=ayeletv
#SBATCH --partition=hns
#SBATCH --array=0-49
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%A_%a.log

set -euo pipefail
ELM=/scratch/users/olgabot/2024-kmerseek-analysis/nextflow-runs/qfo-pfam-region-benchmark/data/elm-motif
IMAGE=/scratch/users/olgabot/apptainer-cache/docker.io-olgabot-kmerseek-0.4.0-rc5.img
export POLARS_MAX_THREADS=${SLURM_CPUS_PER_TASK}

apptainer exec --bind /scratch "$IMAGE" python3 code/scripts/reduce_elm_landing.py \
    --results "$ELM/results" \
    --elm-dir "$ELM" \
    --qfo-bin code/nextflow-runs/qfo-pfam-region-benchmark/bin \
    --notebooks code/notebooks \
    --out-dir landing \
    --task "${SLURM_ARRAY_TASK_ID}" \
    --n-tasks "${SLURM_ARRAY_TASK_COUNT}"
