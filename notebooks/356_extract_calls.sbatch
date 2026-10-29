#!/bin/bash
# SLURM array for scripts/extract_356_ortholog_calls.py (notebook 356, before Stop 0).
#
# Cuts every chicken and zebrafish call table of the midi-plus run (816 kmerseek tables,
# 16 comparison-tool tables) down to the 1:1 ortholog pairs, and writes the AlphaFold
# pLDDT of each pair's proteins. Submit from the work directory, which holds code/ and logs/:
#
#   cd /scratch/users/olgabot/356-exon-landing && sbatch code/notebooks/356_extract_calls.sbatch
#
# A task skips any output that already exists, so resubmitting after a failure only redoes
# what is missing.
#
# Resources from the nb 244 reducer on the same tables (array 45021171, 100 tasks, 4 CPUs):
# peak 33.5 GB against a 32 GB ask, under 2 min per task. This array reads each table once
# and keeps far fewer rows, so 48 GB and 2 h leave room without a retry.
#SBATCH --job-name=356-extract
#SBATCH --account=ayeletv
#SBATCH --partition=hns
#SBATCH --array=0-99
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%A_%a.log

set -euo pipefail
MIDI=/scratch/users/olgabot/2024-kmerseek-analysis/nextflow-runs/qfo-pfam-region-benchmark/data/midi-plus
IMAGE=/scratch/users/olgabot/apptainer-cache/docker.io-olgabot-kmerseek-0.4.0-rc5.img
WORK=$(pwd)
export POLARS_MAX_THREADS=${SLURM_CPUS_PER_TASK}

apptainer exec --bind /scratch "$IMAGE" python3 code/scripts/extract_356_ortholog_calls.py \
    --results "$MIDI/results" \
    --structures "$MIDI/structures" \
    --qfo-bin code/nextflow-runs/qfo-pfam-region-benchmark/bin \
    --pairs code/notebooks/356_data/orthologs_1to1.tsv \
    --out-dir "$WORK/out" \
    --task "${SLURM_ARRAY_TASK_ID}" \
    --n-tasks "${SLURM_ARRAY_TASK_COUNT}"
