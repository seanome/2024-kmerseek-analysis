#!/bin/bash
#SBATCH --job-name=oyster-stopfree-full-DE-pilot
#SBATCH --partition=normal,hns
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
set -euo pipefail
export PATH="$GROUP_HOME/opt/oyster-shazam-20261023/bin:$GROUP_HOME/opt/oyster-mvp/sw/TrimGalore-0.6.10:$PATH"
"$GROUP_HOME/opt/oyster-shazam-20261023/bin/python" "$HOME/oyster-shazam-code/stopfree_public_pilot.py"
