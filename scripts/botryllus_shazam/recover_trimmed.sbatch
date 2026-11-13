#!/bin/bash
#SBATCH --job-name=botryllus-validate-trimmed
#SBATCH --partition=normal,hns
#SBATCH --cpus-per-task=12
#SBATCH --mem=8G
#SBATCH --time=04:00:00
set -euo pipefail
ROOT="$SCRATCH/botryllus-shazam-PEC-20261023"
CODE="${1:?pass versioned recovery code directory}"
ml devel rust/1.90.0
PY="$GROUP_HOME/opt/oyster-shazam-20261023/bin/python"
OUT="$ROOT/trimmed-recovery-${SLURM_JOB_ID}"
mkdir "$OUT"
rustc --test "$CODE/validate_pairs.rs" -o "$L_SCRATCH_JOB/validator-tests"
"$L_SCRATCH_JOB/validator-tests"
rustc -O "$CODE/validate_pairs.rs" -o "$L_SCRATCH_JOB/validate_pairs"
sha256sum "$CODE/stage_trimmed.py" "$CODE/validate_pairs.rs" "$CODE/ena-runs.tsv" > "$OUT/code.sha256"
/usr/bin/time -v "$PY" "$CODE/stage_trimmed.py" "$ROOT" "$CODE/ena-runs.tsv" \
    "$L_SCRATCH_JOB/validate_pairs" --workers 4 > "$OUT/validation.log" 2> "$OUT/validation.time.log"
/bin/cp "$ROOT/trimmed-inputs.validated.json" "$OUT/trimmed-inputs.validated.json"
