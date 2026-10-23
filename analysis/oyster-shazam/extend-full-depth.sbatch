#!/bin/bash
#SBATCH --job-name=oyster-full-all-significant
#SBATCH --partition=normal,hns
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
set -euo pipefail
ROOT="$SCRATCH/oyster-full-extensions-20261023"
"$GROUP_HOME/opt/oyster-shazam-20261023/bin/python" "$HOME/oyster-shazam-code/extend_all_significant.py" --root "$ROOT" --batch "$SLURM_ARRAY_TASK_ID" --scores "$SCRATCH/oyster-mvp-20261023/splash/full-memory-corrected/oyster.after_correction.scores.tsv" --reads "$ROOT/reads.txt" --compactors "$GROUP_HOME/opt/oyster-mvp/sw/splash-2.11.9/compactors"
