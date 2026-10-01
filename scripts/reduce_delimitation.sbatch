#!/bin/bash
# SLURM array for scripts/delimitation_pre_transfer.py --results (notebook 238).
#
# Scores every kmerseek region parquet and every comparison-tool TSV of the ten midi-plus
# targets against the human Pfam instances. Each task takes every n-th file by name.
# Submitted by `make reduce-delimitation` in nextflow-runs/qfo-pfam-region-benchmark, from a
# work directory holding code/ (a git archive of HEAD) and logs/; that target also passes
# the account, partition and array size, and queues the job that joins the outputs.
#
# Memory and time: on the Mac a 2.9 GB region file (19.4 million significant regions) took
# 3 s at 5.1 GB peak, and the largest here is 4.5 GB. HHblits vs mouse took 12 s at 7.3 GB
# on Sherlock.
#
# A task skips any file whose domains parquet already exists, so resubmitting the same array
# after a failure redoes only what is missing.
#SBATCH --job-name=238-delimitation
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
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
