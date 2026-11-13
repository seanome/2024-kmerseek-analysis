#!/bin/bash
# Notebook 342, Stop 3: rerun the scoring after job 48685827 (the early score of three settings)
# was killed at 48 GB. The scorer now reads one call file at a time and keeps only the best
# call per exon (commit 59762371); measured at 40.5 GB on the largest file (job 48687067), so
# scoring asks the 64 GB in tables/342/342_stop3_resources.tsv.
#
#   1. Resubmit the early score of the three finished settings into $WORK/score_3settings.
#   2. Raise the pending four-setting score job (from scripts/342_stop3_protein20_resubmit.sh)
#      to 64 GB. If Slurm refuses the change, cancel it and submit it again with the same
#      afterok dependency on the protein20 search.
#
# Run from the repository root of a checkout of olgabot/exon-function-342 on a Sherlock login
# node. Job ids are appended to $WORK/stop3_protein20_jobs.txt.
set -euo pipefail

TABLE=tables/342/342_stop3_resources.tsv
SB=scripts/342_stop3.sbatch
[ -f "$TABLE" ] && [ -f "$SB" ] || { echo "run from the repository root" >&2; exit 2; }
WORK=${NB342_WORK:-$SCRATCH/exon-function-342}
JOBS=$WORK/stop3_protein20_jobs.txt
# Two early scores write the same files. On 2026-10-20 this script ran twice and a second early
# score (48690800) ran next to 48689195 until it was cancelled, so refuse while one is queued.
running=$(squeue -h -u "$USER" -n 342-score-3settings -o %i 2>/dev/null || true)
[ -z "$running" ] || { echo "342: early score job $running is still queued or running; not resubmitting" >&2; exit 3; }
read -r cpus mem hours < <(awk -F'\t' '!/^#/ && $1 == "score" { print $5, $6, $7 }' "$TABLE")
res="--cpus-per-task=$cpus --mem=${mem}G --time=$(printf %02d "$hours"):00:00"

early=$(sbatch --parsable --job-name=342-score-3settings $res \
    --export=ALL,STEP=score,CALLS_DIR=$WORK/calls_3settings,SCORE_OUT=$WORK/score_3settings "$SB")
printf 'score-3settings (rerun)\t%s\n' "$early" | tee -a "$JOBS"

score=$(awk -F'\t' '$1 == "score" { id = $2 } END { print id }' "$JOBS")
search=$(awk -F'\t' '$1 == "search-protein20" { id = $2 } END { print id }' "$JOBS")
state=$(squeue -h -j "$score" -o %T 2>/dev/null || true)
if [ "$state" = PENDING ] && scontrol update JobId="$score" MinMemoryNode=$((mem * 1024)) TimeLimit="$(printf %02d "$hours"):00:00"; then
    echo "score $score: now ${mem}G, ${hours} h"
else
    [ -n "$state" ] && scancel "$score"
    score=$(sbatch --parsable --job-name=342-score $res --dependency=afterok:"$search" \
        --kill-on-invalid-dep=yes --export=ALL,STEP=score "$SB")
    printf 'score\t%s\n' "$score" | tee -a "$JOBS"
fi
