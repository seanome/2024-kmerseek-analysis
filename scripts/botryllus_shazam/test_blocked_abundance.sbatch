#!/bin/bash
#SBATCH --job-name=botryllus-blocked-model-test
#SBATCH --partition=hns,normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
set -euo pipefail
ROOT="$SCRATCH/botryllus-shazam-PEC-20261023"
CODE="$HOME/botryllus-shazam-code/dekupl-20261024"
PREFIX="$GROUP_HOME/opt/botryllus-dekupl-1.3.3-20261024"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
"$PREFIX/bin/Rscript" "$CODE/test_blocked_abundance.R" "$ROOT/dekupl-setup/blocked-model-check" "$CODE"
