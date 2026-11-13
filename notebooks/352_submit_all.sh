#!/bin/bash
# Notebook 352: submit the whole measurement at once; every job waits on the one before,
# so nothing has to be started by hand. Run on a Sherlock login node from the repo root:
#
#   bash notebooks/352_submit_all.sh
#
#   setup ─► 1 billion residues ─► 10 billion residues ─► gate ─► third size (100, 50,
#                                                                 30 or 20 billion, or none)
#
# Time limits (predictions; how they were made is in the PR and in each script's header):
# setup 12 h, 1 billion 4 h, 10 billion 16 h (predicted 0.3 h and 3.5 h, scaled linearly
# from the dark-set Swiss-Prot runs; the PR has the table). Each size job also stops starting queries
# when the 48 h budget runs out. Job ids go to $SCRATCH/352-kmerseek-scale/job_ids.txt;
# follow them with
#   squeue --me -n 352-setup,352-size,352-gate
set -euo pipefail

cd "$(dirname "$0")/.."
WORK=${NB352_WORK:-$SCRATCH/352-kmerseek-scale}
mkdir -p "$WORK"
SUB=(sbatch --parsable --kill-on-invalid-dep=yes)

SETUP=$("${SUB[@]}" notebooks/352_setup.sbatch)
SETUP=${SETUP%%;*}
S1=$("${SUB[@]}" --dependency=afterok:"$SETUP" --export=ALL,SIZE=1000000000 --time=04:00:00 notebooks/352_size.sbatch)
S1=${S1%%;*}
S10=$("${SUB[@]}" --dependency=afterok:"$S1" --export=ALL,SIZE=10000000000 --time=16:00:00 notebooks/352_size.sbatch)
S10=${S10%%;*}
GATE=$("${SUB[@]}" --dependency=afterok:"$S10" notebooks/352_gate.sbatch)
GATE=${GATE%%;*}
printf '%s submitted from %s at %s\nsetup %s\nsize 1000000000 %s\nsize 10000000000 %s\ngate %s\n' \
    "$(git rev-parse --short HEAD)" "$PWD" "$(date '+%F %T')" "$SETUP" "$S1" "$S10" "$GATE" \
    | tee -a "$WORK/job_ids.txt"
