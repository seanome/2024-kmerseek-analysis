#!/bin/bash
# SLURM array for scripts/delimitation_pre_transfer.py --results (notebook 238).
#
# Scores every kmerseek region parquet and every comparison-tool TSV of the ten midi-plus
# targets against the human Pfam instances. Each task takes every n-th file by name.
# Submitted by `make reduce-delimitation` in nextflow-runs/qfo-pfam-region-benchmark, from a
# work directory holding code/ (a git archive of HEAD) and logs/; that target also passes
# the account, partition and array size, and queues the job that joins the outputs.
#
# A task skips any file whose domains parquet already exists, so resubmitting the same array
# after a failure redoes only what is missing.
#SBATCH --job-name=238-delimitation
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%A_%a.log

set -euo pipefail
: "${DELIM_RESULTS:?set by make reduce-delimitation}"
: "${DELIM_IMAGE:?set by make reduce-delimitation}"
export POLARS_MAX_THREADS=${SLURM_CPUS_PER_TASK}

apptainer exec --bind /scratch "$DELIM_IMAGE" python3 code/scripts/delimitation_pre_transfer.py \
    --results "$DELIM_RESULTS" \
    --truth "$DELIM_RESULTS/truth/human_domain_truth.parquet" \
    --out-dir per-job \
    ${DELIM_ONLY:+--only "$DELIM_ONLY"} \
    --task "${SLURM_ARRAY_TASK_ID}" \
    --n-tasks "${SLURM_ARRAY_TASK_COUNT}"
