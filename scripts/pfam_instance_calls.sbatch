#!/bin/bash
# SLURM array for scripts/pfam_instance_calls.py (stage 1 of the Pfam candidate table).
#
# Reads every mask-on kmerseek setting and six comparison tools for the nine
# midi-plus species, and writes one small per-instance parquet per (setting or tool, species)
# under pfam_calls/. Submit from the work directory, which holds code/ (git archive) and logs/:
#
#   cd $SCRATCH/pfam-candidates && sbatch code/scripts/pfam_instance_calls.sbatch
#
# A task skips any (setting, species) whose output already exists, so resubmitting the same
# array after a failure redoes only what is missing. Sized from notebook 244's
# reduce_swissprot_instance_landing array (job 45021171), which read the same files: its
# peak was 32.0 GB, exactly its 32G ask, and its longest task took 2 minutes.
#SBATCH --job-name=pfam-calls
#SBATCH --account=ayeletv
#SBATCH --partition=hns
#SBATCH --array=0-99
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%A_%a.log

set -euo pipefail
RESULTS=/scratch/users/olgabot/2024-kmerseek-analysis/nextflow-runs/qfo-pfam-region-benchmark/data/midi-plus/results
IMAGE=/scratch/users/olgabot/apptainer-cache/docker.io-olgabot-kmerseek-0.4.0-rc5.img
export POLARS_MAX_THREADS=${SLURM_CPUS_PER_TASK}

apptainer exec --bind /scratch "$IMAGE" python3 code/scripts/pfam_instance_calls.py \
    --results "$RESULTS" \
    --qfo-bin code/nextflow-runs/qfo-pfam-region-benchmark/bin \
    --out-dir pfam_calls \
    --task "${SLURM_ARRAY_TASK_ID}" \
    --n-tasks "${SLURM_ARRAY_TASK_COUNT}"
