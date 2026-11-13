#!/bin/bash
#SBATCH --job-name=botryllus-PEC-download-recovery
#SBATCH --partition=hns,normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=24:00:00
#SBATCH --array=1,5,6%3
set -euo pipefail
CODE="$HOME/botryllus-shazam-code/recovery-md5-20261023"
ROOT="$SCRATCH/botryllus-shazam-PEC-20261023"
ml devel python/3.12.1 rust/1.90.0
mkdir -p "$L_SCRATCH_JOB/bin"
rustc -O "$CODE/validate_pairs.rs" -o "$L_SCRATCH_JOB/bin/validate_pairs"
sha256sum "$CODE/download.py" "$CODE/validate_pairs.rs" "$CODE/ena-runs.tsv"
/usr/bin/time -v python3 "$CODE/download.py" "$CODE/ena-runs.tsv" "$ROOT" \
    "$L_SCRATCH_JOB/bin/validate_pairs" --batch "$SLURM_ARRAY_TASK_ID"
