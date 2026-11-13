#!/bin/bash
#SBATCH --job-name=botryllus-storage-Branchwater
#SBATCH --partition=hns,normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
set -euo pipefail
"$GROUP_HOME/opt/oyster-shazam-20261023/bin/python" "$HOME/botryllus-shazam-code/storage_benchmark.py" "$SCRATCH/botryllus-shazam-PEC-20261023"
