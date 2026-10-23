#!/bin/bash
#SBATCH --job-name=oyster-all-significant
#SBATCH --partition=hns,normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-21%2
set -euo pipefail
ml devel python/3.12.1
python3 "$HOME/oyster-shazam-code/extend_all_significant.py" --root "$SCRATCH/oyster-shazam-20261023" --batch "$SLURM_ARRAY_TASK_ID"
