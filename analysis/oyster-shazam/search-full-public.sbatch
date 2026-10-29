#!/bin/bash
#SBATCH --job-name=oyster-full-public-search
#SBATCH --partition=normal,hns
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-553%2
set -euo pipefail
export PATH="$GROUP_HOME/opt/oyster-shazam-20261023/bin:$PATH"
"$GROUP_HOME/opt/oyster-shazam-20261023/bin/python" "$HOME/oyster-shazam-code/search_extended.py" --root "$SCRATCH/oyster-full-extensions-20261023" --batch "$SLURM_ARRAY_TASK_ID" --reference-dir "$SCRATCH/oyster-shazam-20261023/public-bivalve-indexes" --output-dir search-public-zero --scales 10 100 --query-prefix splash:full
