#!/bin/bash
#SBATCH --job-name=botryllus-DEkupl-read-extension-manysearch
#SBATCH --partition=hns,normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=48:00:00
set -euo pipefail
ml devel rust/1.90.0
ROOT="$SCRATCH/botryllus-shazam-PEC-20261023"
CODE="$HOME/botryllus-shazam-code/dekupl-read-search-20261025"
"$GROUP_HOME/opt/oyster-shazam-20261023/bin/python" "$CODE/dekupl_read_search.py" "$ROOT" "$CODE" "$ROOT/dekupl-read-search-20261025"
