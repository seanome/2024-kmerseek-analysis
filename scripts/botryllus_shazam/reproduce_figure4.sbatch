#!/bin/bash
#SBATCH --job-name=botryllus-Fig4-supplied-count-refit
#SBATCH --partition=hns,normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
set -euo pipefail
ROOT="$SCRATCH/botryllus-shazam-PEC-20261023"
CODE="$HOME/botryllus-shazam-code/figure4-20261024"
PREFIX="$GROUP_HOME/opt/botryllus-dekupl-1.3.3-20261024"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
sha256sum "$CODE/reproduce_figure4.R" "$ROOT/figure4-inputs/published-counts.tsv.gz"
"$PREFIX/bin/Rscript" "$CODE/reproduce_figure4.R" "$ROOT/figure4-inputs" "$ROOT/figure4-refit-20261024" "$CODE/abundance-design-20261024.json"
