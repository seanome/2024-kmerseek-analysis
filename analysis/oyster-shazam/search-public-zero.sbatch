#!/bin/bash
#SBATCH --job-name=oyster-public-explicit-zero
#SBATCH --partition=normal,hns
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-21%2
set -euo pipefail
export PATH="$GROUP_HOME/opt/oyster-shazam-20261023/bin:$PATH"
"$GROUP_HOME/opt/oyster-shazam-20261023/bin/python" "$HOME/oyster-shazam-code/search_extended.py" --root "$SCRATCH/oyster-shazam-20261023" --batch "$SLURM_ARRAY_TASK_ID" --reference-dir public-bivalve-indexes --output-dir search-public-bivalve-zero --scales 10 100
