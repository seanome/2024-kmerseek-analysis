#!/bin/bash
#SBATCH --job-name=oyster-full-extension-audit
#SBATCH --partition=normal,hns
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
set -euo pipefail
"$GROUP_HOME/opt/oyster-shazam-20261023/bin/python" "$HOME/oyster-shazam-code/audit_full_extensions.py" "$SCRATCH/oyster-full-extensions-20261023"
