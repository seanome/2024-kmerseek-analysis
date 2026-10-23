#!/bin/bash
# Notebook 353, development half: one array task per candidate setting (ten, fixed in
# notebooks/353_prereg.md). Each builds its index and searches 36,000 self-retrieval
# queries plus about 5,600 dipeptide shuffles.
#
#   cd /scratch/users/olgabot/2024-kmerseek-analysis-nb353 && sbatch scripts/353/dev_array.sbatch
#SBATCH --job-name=nb353-dev
#SBATCH --account=ayeletv
#SBATCH --partition=hns,normal
#SBATCH --array=0-9
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/users/olgabot/nb353/logs/%x.%A_%a.out
set -euo pipefail
SETTINGS=(dayhoff6:5 dayhoff6:6 dayhoff6:7 dayhoff6:8 sdm12:4 sdm12:5 sdm12:6 mmseqs12:4 mmseqs12:5 mmseqs12:6)
s=${SETTINGS[$SLURM_ARRAY_TASK_ID]}
cd "${SLURM_SUBMIT_DIR}"
bash scripts/353/run_setting.sh dev "${s%%:*}" "${s##*:}" /scratch/users/olgabot/nb353
