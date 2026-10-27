#!/bin/bash
#SBATCH --job-name=botryllus-production-preflight
#SBATCH --partition=normal,hns
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
set -euo pipefail
ml biology hmmer/3.4 minimap2/2.30
CODE=${1:?versioned source}
"$GROUP_HOME/opt/oyster-shazam-20261023/bin/python" "$CODE/scripts/botryllus_shazam/production_preflight.py" "$CODE" "$SCRATCH/botryllus-production-preflight-${SLURM_JOB_ID}"
