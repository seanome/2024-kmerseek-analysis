#!/bin/bash
#SBATCH --job-name=botryllus-Fig4-SPLASH-DEkupl-recovery
#SBATCH --partition=hns,normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
set -euo pipefail
ROOT="$SCRATCH/botryllus-shazam-PEC-20261023"
CODE="$HOME/botryllus-shazam-code/figure4-20261024"
"$GROUP_HOME/opt/oyster-shazam-20261023/bin/python" "$CODE/figure4_recovery.py" "$ROOT" "$ROOT/figure4-inputs" "$ROOT/figure4-recovery-20261024"
