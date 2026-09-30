#!/bin/bash
# One SLURM task for scripts/pfam_fair_table.py (stage 2): after every stage-1 task has
# finished, build the unfiltered table, choose kmerseek's setting per length bin and per clan
# on the selection half, and write the held-out summary and candidate views to pfam_out/.
#
#   cd $SCRATCH/pfam-candidates && sbatch --dependency=afterok:<stage-1 job id> \
#       code/scripts/pfam_fair_table.sbatch
#
# Pfam-A.clans.tsv.gz must come from the same Pfam release as the run's truth; fetch it on a
# login node (compute nodes have no internet) and put it next to code/. The truth is Pfam
# 38.2 (notebooks/fetch_pfam_envelope_coords.py), so fetch it from releases/Pfam38.2/, not
# current_release. This stage reads at most 1,870 files of up to 2,435 rows each.
#SBATCH --job-name=pfam-table
#SBATCH --account=ayeletv
#SBATCH --partition=hns
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.log

set -euo pipefail
RESULTS=/scratch/users/olgabot/2024-kmerseek-analysis/nextflow-runs/qfo-pfam-region-benchmark/data/midi-plus/results
IMAGE=/scratch/users/olgabot/apptainer-cache/docker.io-olgabot-kmerseek-0.4.0-rc5.img
export POLARS_MAX_THREADS=${SLURM_CPUS_PER_TASK}

apptainer exec --bind /scratch "$IMAGE" python3 code/scripts/pfam_fair_table.py \
    --calls-dir pfam_calls \
    --truth "$RESULTS/truth/human_domain_truth.parquet" \
    --clans Pfam-A.clans.tsv.gz \
    --out-dir pfam_out
