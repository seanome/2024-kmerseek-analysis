#!/bin/bash
#SBATCH --job-name=botryllus-manysearch-comparison
#SBATCH --partition=hns,normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
set -euo pipefail
"$GROUP_HOME/opt/oyster-shazam-20261023/bin/python" "$HOME/botryllus-shazam-code/compare_manysearch.py" "$SCRATCH/botryllus-shazam-PEC-20261023"
