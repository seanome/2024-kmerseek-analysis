#!/bin/bash
#SBATCH --job-name=botryllus-preflight-repair
#SBATCH --partition=normal,hns
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
set -euo pipefail
CODE=${1:?immutable source}
PREVIOUS=${2:?preserved preflight}
PY="$GROUP_HOME/opt/oyster-shazam-20261023/bin/python"
"$PY" "$CODE/scripts/botryllus_shazam/recover_production_preflight.py" "$CODE" "$PREVIOUS" "$SCRATCH/botryllus-production-preflight-${SLURM_JOB_ID}"
bash "$CODE/nextflow-runs/botryllus-shazam/smoke.sbatch" "$CODE"
