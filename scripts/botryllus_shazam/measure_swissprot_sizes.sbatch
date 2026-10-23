#!/bin/bash
#SBATCH --job-name=botryllus-SwissProt-size-comparison
#SBATCH --partition=hns,normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=00:30:00
set -euo pipefail
ROOT="$SCRATCH/botryllus-shazam-PEC-20261023"
"$GROUP_HOME/opt/oyster-shazam-20261023/bin/python" "$HOME/botryllus-shazam-code/measure_swissprot_sizes.py" "$ROOT"
