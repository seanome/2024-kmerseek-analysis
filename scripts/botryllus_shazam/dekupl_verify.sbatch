#!/bin/bash
#SBATCH --job-name=botryllus-DEkupl-software-checks
#SBATCH --partition=hns,normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
set -euo pipefail
ROOT="$SCRATCH/botryllus-shazam-PEC-20261023"
CODE="$HOME/botryllus-shazam-code/dekupl-20261024"
PREFIX="$GROUP_HOME/opt/botryllus-dekupl-1.3.3-20261024"
export PATH="$PREFIX/bin:$PATH"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
sha256sum "$CODE"/*.py "$CODE"/*.R
"$GROUP_HOME/opt/oyster-shazam-20261023/bin/python" "$CODE/dekupl_smoke.py" "$ROOT" "$PREFIX" "$ROOT/dekupl-setup/smoke-v2"
"$PREFIX/bin/Rscript" "$CODE/test_blocked_abundance.R" "$ROOT/dekupl-setup/blocked-model-check" "$CODE" "$PREFIX"
