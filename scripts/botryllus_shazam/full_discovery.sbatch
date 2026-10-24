#!/bin/bash
#SBATCH --job-name=botryllus-full-SPLASH-all-anchors
#SBATCH --partition=hns,normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=48:00:00
set -euo pipefail
ml devel python/3.12.1
ROOT="$SCRATCH/botryllus-shazam-PEC-20261023"
CODE="$HOME/botryllus-shazam-code/full-discovery-20261023"
PYTHON="$GROUP_HOME/opt/oyster-shazam-20261023/bin/python"
SOFTWARE="$SCRATCH/oyster-shazam-20261023/software"
"$PYTHON" "$CODE/full_discovery.py" "$ROOT" "$SOFTWARE" "$ROOT/full-depth-discovery-20261023"
