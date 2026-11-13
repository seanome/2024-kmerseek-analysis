#!/bin/bash
#SBATCH --job-name=nb361-freeze
#SBATCH --partition=hns,normal
#SBATCH --account=ayeletv
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=%x.%j.out
# Sample-to-contrast table from SI Dataset S1 + the ENA report. Its output is copied into the
# repository (design/) and committed before any contrast is run.
set -euo pipefail
: "${PEC_ROOT:?}" "${PEC_CODE:?}"
cd "$PEC_ROOT"
"$PEC_ROOT/env/bin/python" "$PEC_CODE/scripts/freeze_design.py" si/pnas.2610968123.sd01.xlsx \
  meta/ena_filereport.tsv design > design_freeze.log 2>&1
cat design_freeze.log
