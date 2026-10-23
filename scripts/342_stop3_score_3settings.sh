#!/bin/bash
# Notebook 342, Stop 3: score the three settings that have E-values (aro_hyd_chg_pol4 k11,
# aro_hyd_arg_chg_pol5 k9, hp_pbotc_1st_ed2 k21) again, with the scorer that drops calls whose
# E is not a finite number (2026-10-21: over 99.9% of every setting's calls had E = inf, and
# the search step's awk filter let them through). Writes $WORK/score_3settings, the input of
# notebook 342 section 3b. protein20 k5 has no E-values in this kmerseek build (its
# closed-form match probability is 0.000), so it is not searched again here.
#
# Run from the repository root of a checkout of olgabot/exon-function-342 on a Sherlock login
# node. The job id is appended to $WORK/stop3_protein20_jobs.txt.
set -euo pipefail

TABLE=tables/342/342_stop3_resources.tsv
SB=scripts/342_stop3.sbatch
[ -f "$TABLE" ] && [ -f "$SB" ] || { echo "run from the repository root" >&2; exit 2; }
WORK=${NB342_WORK:-$SCRATCH/exon-function-342}
queued=$(squeue -h -u "$USER" -n 342-score,342-score-3settings -o %i 2>/dev/null || true)
[ -z "$queued" ] || { echo "342: score jobs still queued or running: $queued; not submitting" >&2; exit 3; }
read -r cpus mem hours < <(awk -F'\t' '!/^#/ && $1 == "score" { print $5, $6, $7 }' "$TABLE")
job=$(sbatch --parsable --job-name=342-score-3settings --cpus-per-task="$cpus" --mem="${mem}G" \
    --time="$(printf %02d "$hours")":00:00 \
    --export=ALL,STEP=score,CALLS_DIR=$WORK/calls_3settings,SCORE_OUT=$WORK/score_3settings "$SB")
printf 'score-3settings (finite E)\t%s\n' "$job" | tee -a "$WORK/stop3_protein20_jobs.txt"
