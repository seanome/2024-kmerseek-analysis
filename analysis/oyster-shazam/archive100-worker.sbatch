#!/bin/bash
#SBATCH --job-name=oyster-bivalve-s10-s100
#SBATCH --partition=hns,normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
set -euo pipefail
ml biology trim_galore/0.6.10
export ARCHIVE_WORKER_HOURS=44
export PATH="$GROUP_HOME/opt/oyster-shazam-20261023/bin:$PATH"
/usr/bin/time -v "$GROUP_HOME/opt/oyster-shazam-20261023/bin/python" "$HOME/oyster-shazam-code/archive100.py" "$SLURM_ARRAY_TASK_ID"
