#!/bin/bash
# Notebook 299: submit the whole pilot at once; every job waits on the ones it needs, so
# nothing has to be started by hand afterwards. Run on a Sherlock login node:
#
#   bash notebooks/299_submit_all.sh
#
#   setup ─► step 1 (18 tasks) ─► gate ─┬─ dead: stops, step 1 verdict written
#                                       └─ fits: DIAMOND + 2 sketch arrays ─► search ─► finish
#
# The gate submits the later jobs itself, because the sketch arrays are sized from step
# 1's measured rates. Job ids go to $SCRATCH/299-omrgc-pilot/job_ids.txt; follow them with
#   squeue --me -n 299-setup,299-step1,299-gate,299-diamond,299-sketch,299-search,299-finish
set -euo pipefail

cd "$(dirname "$0")/.."
WORK=${OMRGC_WORK:-$SCRATCH/299-omrgc-pilot}
mkdir -p "$WORK"
SUB=(sbatch --parsable --kill-on-invalid-dep=yes)

SETUP=$("${SUB[@]}" notebooks/299_setup.sbatch)
SETUP=${SETUP%%;*}
STEP1=$("${SUB[@]}" --dependency=afterok:"$SETUP" notebooks/299_step1_sketch_timing.sbatch)
STEP1=${STEP1%%;*}
GATE=$("${SUB[@]}" --dependency=afterok:"$STEP1" notebooks/299_gate.sbatch)
GATE=${GATE%%;*}
printf '%s submitted from %s at %s\nsetup %s\nstep1 %s\ngate %s\n' \
    "$(git rev-parse --short HEAD)" "$PWD" "$(date '+%F %T')" "$SETUP" "$STEP1" "$GATE" \
    | tee -a "$WORK/job_ids.txt"
