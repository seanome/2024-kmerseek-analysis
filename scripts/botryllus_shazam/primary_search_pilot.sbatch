#!/bin/bash
#SBATCH --job-name=botryllus-Sourmash-primary-pilot
#SBATCH --partition=hns,normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
set -euo pipefail
ROOT="$SCRATCH/botryllus-shazam-PEC-20261023"
PYTHON="$GROUP_HOME/opt/oyster-shazam-20261023/bin/python"
CODE="$HOME/botryllus-shazam-code/primary-search-20261023"
OUT="$ROOT/primary-search-pilot-20261023"
mkdir "$OUT"
"$PYTHON" "$CODE/primary_search.py" "$ROOT" "$OUT" "$ROOT/checkpoints/pilot/results/peptides.fasta"
