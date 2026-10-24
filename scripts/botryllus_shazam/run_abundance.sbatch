#!/bin/bash
#SBATCH --job-name=botryllus-DEkupl-gene-DE
#SBATCH --partition=hns,normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=48:00:00
set -euo pipefail
ml devel python/3.12.1
ROOT="$SCRATCH/botryllus-shazam-PEC-20261023"
CODE="$HOME/botryllus-shazam-code/dekupl-20261024"
PREFIX="$GROUP_HOME/opt/botryllus-dekupl-1.3.3-20261024"
export PATH="$PREFIX/bin:$PATH"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
sha256sum "$CODE"/*.py "$CODE"/*.R "$CODE/abundance-design-20261024.json" > "$ROOT/logs/abundance-code.${SLURM_JOB_ID}.sha256"
df -h "$L_SCRATCH_JOB"
"$GROUP_HOME/opt/oyster-shazam-20261023/bin/python" "$CODE/run_abundance.py" "$ROOT" "$CODE" "$PREFIX"
