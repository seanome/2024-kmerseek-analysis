#!/bin/bash
# Notebook 353, held-out half: the frozen setting (data/kill_tests/short_stretch/
# frozen_setting.json, committed before this job is submitted) on every held-out record:
# about 204,000 self-retrieval queries and 657,000 shuffles.
#
#   cd /scratch/users/olgabot/2024-kmerseek-analysis-nb353 && sbatch scripts/353/held_out.sbatch
#SBATCH --job-name=nb353-heldout
#SBATCH --account=ayeletv
#SBATCH --partition=hns,normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/users/olgabot/nb353/logs/%x.%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
frozen=data/kill_tests/short_stretch/frozen_setting.json
alphabet=$(python3 -c "import json;print(json.load(open('$frozen'))['alphabet'])")
k=$(python3 -c "import json;print(json.load(open('$frozen'))['k'])")
git log -1 --format='%H %s' -- "$frozen"
bash scripts/353/run_setting.sh held_out "$alphabet" "$k" /scratch/users/olgabot/nb353
