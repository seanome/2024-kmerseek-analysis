#!/bin/bash
# SLURM array for scripts/reduce_elm_cover.py (notebook 250, ELM cover search).
#
# Scores every arm of one target species: 268 kmerseek arms and the eight comparison tools
# for chicken. Each task takes every 20th arm by name. Submitted by `make reduce-elm-cover`
# in nextflow-runs/qfo-pfam-region-benchmark, from a work directory holding code/ (a git
# archive of HEAD) and logs/. ELM_SPECIES picks the target (default chicken).
#
# A task skips any arm whose summary JSON already exists, so resubmitting the same array
# after a failure redoes only what is missing. On the Mac, three arms (9 MB to 1.6 GB of
# parquet) took 3.5 s at 3.4 GB peak; the largest arms here are 24 GB and 583 million rows,
# hence 64 GB.
#SBATCH --job-name=250-elm-cover
#SBATCH --account=ayeletv
#SBATCH --partition=hns,normal
#SBATCH --array=0-19
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%A_%a.log

set -euo pipefail
PIPE=/scratch/users/olgabot/2024-kmerseek-analysis/nextflow-runs/qfo-pfam-region-benchmark
COVER=$PIPE/data/elm-cover
IMAGE=/scratch/users/olgabot/apptainer-cache/docker.io-olgabot-kmerseek-0.4.0-rc5.img
SPECIES=${ELM_SPECIES:-chicken}
ASSETS=code/nextflow-runs/qfo-pfam-region-benchmark/assets
export POLARS_MAX_THREADS=${SLURM_CPUS_PER_TASK}

# Comparator tables are unzipped next to the task before polars reads them; one scratch
# folder per task keeps two tasks from writing the same file.
mkdir -p "tmp/${SLURM_ARRAY_TASK_ID}"
cd "tmp/${SLURM_ARRAY_TASK_ID}"
apptainer exec --bind /scratch "$IMAGE" python3 ../../code/scripts/reduce_elm_cover.py \
    --results "$COVER/results" \
    --species "$SPECIES" \
    --instances "../../$ASSETS/elm_cover_instances.tsv" \
    --orthologs "../../$ASSETS/elm_cover_orthologs.tsv" \
    --projections "../../$ASSETS/elm_cover_projections.tsv" \
    --query-fasta "$COVER/qfo/Eukaryota/UP000005640_9606.fasta" \
    --structures "$COVER/structures/human" \
    --qfo-bin ../../code/nextflow-runs/qfo-pfam-region-benchmark/bin \
    --out-dir "../../cover/$SPECIES" \
    --task "${SLURM_ARRAY_TASK_ID}" \
    --n-tasks "${SLURM_ARRAY_TASK_COUNT}"
