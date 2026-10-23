#!/bin/bash
# Notebook 299: add scaled 10 (settings 18-26) to a pilot whose scaled 100 and 1000 chain is
# already submitted. One command on a Sherlock login node; every job waits on what it needs:
#
#   bash notebooks/299_submit_scaled10.sh [FINISH_JOB_OF_THE_RUNNING_CHAIN]
#
#   step 1, settings 18-26 ─► gate (ONLY_SCALED=10) ─┬─ scaled 10 over budget: stops
#                                                    └─ fits: sketch s10 ─► search 18-26 ─► finish
#
# The final finish also waits for the given job, so the last scoring sees all 27 settings.
# The setup job is not rerun: the catalog, slice and parts are already in the work folder.
set -euo pipefail

cd "$(dirname "$0")/.."
WORK=${OMRGC_WORK:-$SCRATCH/299-omrgc-pilot}
AFTER=${1:-}
SUB=(sbatch --parsable --kill-on-invalid-dep=yes)
[ -s "$WORK/catalog_stats.json" ] || { echo "no $WORK/catalog_stats.json: run 299_submit_all.sh first"; exit 1; }

# 40 G: at scaled 10 the sketch step reached the 8 G limit (7.8-8.0 GB peak, three settings
# killed inside it; array 48668306). Scaled 100 peaked at 2.6 GB, and scaled 10 keeps about
# 10 times the hashes, so about 26 GB; 40 G leaves room.
STEP1=$("${SUB[@]}" --mem=40G --array=18-26 notebooks/299_step1_sketch_timing.sbatch)
STEP1=${STEP1%%;*}
GATE=$("${SUB[@]}" --dependency=afterok:"$STEP1" --export=ALL,ONLY_SCALED=10,AFTER_FINISH="$AFTER" \
    notebooks/299_gate.sbatch)
GATE=${GATE%%;*}
printf 'scaled 10 submitted from %s at %s: step1 %s (array 18-26), gate %s, final finish also waits for %s\n' \
    "$(git rev-parse --short HEAD)" "$(date '+%F %T')" "$STEP1" "$GATE" "${AFTER:-nothing}" | tee -a "$WORK/job_ids.txt"
