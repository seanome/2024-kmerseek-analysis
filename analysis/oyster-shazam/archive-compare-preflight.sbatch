#!/bin/bash
#SBATCH --job-name=oyster-bivalve-s100-check
#SBATCH --partition=hns,normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
set -euo pipefail
ml biology trim_galore/0.6.10
export PATH="$GROUP_HOME/opt/oyster-shazam-20261023/bin:$PATH"
trim_galore --version
sourmash --version
PYTHON="$GROUP_HOME/opt/oyster-shazam-20261023/bin/python"
"$PYTHON" "$HOME/oyster-shazam-code/archive_compare.py" prepare
"$PYTHON" "$HOME/oyster-shazam-code/archive_compare.py" DRR350416
