#!/bin/bash
#SBATCH --job-name=botryllus-chordate-inventory
#SBATCH --partition=hns,normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
set -euo pipefail
ml devel python/3.12.1
ROOT="$SCRATCH/botryllus-shazam-PEC-20261023"
/usr/bin/time -v python3 "$HOME/botryllus-shazam-code/archive_inventory.py" \
    "$ROOT/chordate-inventory-attempt2-20261023" --scope chordate
