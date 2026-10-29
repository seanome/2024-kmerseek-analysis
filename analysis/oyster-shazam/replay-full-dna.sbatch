#!/bin/bash
#SBATCH --job-name=oyster-full-DE-public-DNA100
#SBATCH --partition=normal,hns
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
set -euo pipefail
export PATH="$GROUP_HOME/opt/oyster-shazam-20261023/bin:$PATH"
"$GROUP_HOME/opt/oyster-shazam-20261023/bin/python" "$HOME/oyster-shazam-code/replay_full_dna.py" search --shard "$SLURM_ARRAY_TASK_ID"
