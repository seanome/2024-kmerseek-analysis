#!/bin/bash
#SBATCH --job-name=botryllus-full-depth-manysearch
#SBATCH --partition=hns,normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
set -euo pipefail
ROOT="$SCRATCH/botryllus-shazam-PEC-20261023"
CODE="$HOME/botryllus-shazam-code/full-search-20261024"
"$GROUP_HOME/opt/oyster-shazam-20261023/bin/python" "$CODE/search_full_depth.py" "$ROOT" "$CODE" "$ROOT/full-depth-manysearch-20261024"
