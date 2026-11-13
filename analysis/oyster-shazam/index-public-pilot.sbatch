#!/bin/bash
#SBATCH --job-name=oyster-public-bivalve-pilot
#SBATCH --partition=hns,normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=06:00:00
set -euo pipefail
export PATH="$GROUP_HOME/opt/oyster-shazam-20261023/bin:$PATH"
/usr/bin/time -v "$GROUP_HOME/opt/oyster-shazam-20261023/bin/python" "$HOME/oyster-shazam-code/index_public_runs.py" "$HOME/oyster-shazam-code/public-bivalve-pilot.tsv" "$SCRATCH/oyster-shazam-20261023"
