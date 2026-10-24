#!/bin/bash
#SBATCH --job-name=botryllus-remaining-inputs
#SBATCH --partition=hns,normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=12:00:00
set -euo pipefail
ml devel python/3.12.1 rust/1.90.0
CODE="$HOME/botryllus-shazam-code/recovery-20261024"
ROOT="$SCRATCH/botryllus-shazam-PEC-20261023"
mkdir -p "$L_SCRATCH_JOB/bin"
rustc -O "$CODE/validate_pairs.rs" -o "$L_SCRATCH_JOB/bin/validate_pairs"
sha256sum "$CODE"/*.py "$CODE/ena-runs.tsv"
/usr/bin/time -v python3 "$CODE/recover_remaining.py" "$ROOT" "$CODE/ena-runs.tsv" "$L_SCRATCH_JOB/bin/validate_pairs"
