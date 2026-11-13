#!/bin/bash
#SBATCH --job-name=botryllus-external-SRA-manysearch
#SBATCH --partition=hns,normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
set -euo pipefail
ml devel rust/1.90.0 biology trim_galore/0.6.10
CODE="$HOME/botryllus-shazam-code/external-search-20261023"
mkdir -p "$L_SCRATCH_JOB/bin"
rustc -O "$CODE/validate_pairs.rs" -o "$L_SCRATCH_JOB/bin/validate_pairs"
"$GROUP_HOME/opt/oyster-shazam-20261023/bin/python" "$CODE/search_external_tunicates.py" "$SCRATCH/botryllus-shazam-PEC-20261023" "$CODE/external-tunicate-pilot.tsv" "$L_SCRATCH_JOB/bin/validate_pairs"
