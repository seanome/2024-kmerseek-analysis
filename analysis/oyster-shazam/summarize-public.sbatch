#!/bin/bash
#SBATCH --job-name=oyster-public-report-check
#SBATCH --partition=hns,normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
set -euo pipefail
"$GROUP_HOME/opt/oyster-shazam-20261023/bin/python" "$HOME/oyster-shazam-code/summarize_public_matches.py"
