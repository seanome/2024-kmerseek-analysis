#!/bin/bash
#SBATCH --job-name=botryllus-PEC-full-download
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=08:00:00
#SBATCH --array=0-7%4
set -euo pipefail
CODE="$HOME/botryllus-shazam-code"
ROOT="$SCRATCH/botryllus-shazam-PEC-20261023"
ml devel python/3.12.1 rust/1.90.0
mkdir -p "$L_SCRATCH_JOB/bin"
rustc -O "$CODE/validate_pairs.rs" -o "$L_SCRATCH_JOB/bin/validate_pairs"
rustc --test "$CODE/validate_pairs.rs" -o "$L_SCRATCH_JOB/bin/validator-tests"
"$L_SCRATCH_JOB/bin/validator-tests"
/usr/bin/time -v python3 "$CODE/download.py" "$CODE/ena-runs.tsv" "$ROOT" \
    "$L_SCRATCH_JOB/bin/validate_pairs" --batch "$SLURM_ARRAY_TASK_ID"
