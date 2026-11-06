#!/bin/bash
#SBATCH --job-name=nb361-kswap-score
#SBATCH --partition=normal,hns
#SBATCH --account=ayeletv
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --output=%x.%j.out
# Scores KaMRaT + DESeq2 with real and swapped labels against the paper's lists, after
# `make run-kamrat-swap` in nextflow-runs/pec-domen2026. Ran as job 51070567, 2026-11-04.
set -euo pipefail
: "${PEC_ROOT:?}" "${PEC_CODE:?}"
"$PEC_ROOT/env/bin/python" "$PEC_CODE/scripts/kamrat_swap_recovery.py" "$PEC_ROOT" \
  "$PEC_CODE/design/label_swap_kamrat/manifest.tsv" "$PEC_ROOT/label_swap/kamrat_score"
