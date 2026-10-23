#!/bin/bash
# Notebook 342, Stop 3, second pass (2026-10-20). The first pass (scripts/342_stop3_submit.sh,
# jobs 48247305-48247328) finished three of the four settings. protein20 k5 did not: its
# whole-protein index refused the Karlin-Altschul fit on 500 queries ("fewer than 4 score bins
# above the peak hold 30 of each", job 48247325), so all 10 searches stopped with exit 4 and
# the score job was cancelled. PR 114 hit the same refusal for protein20 k5 and fitted on
# 2_500 queries instead; this script does the same. It submits, each with an afterok
# dependency on the jobs it reads:
#
#   score-3settings   kind 4 features from the three finished settings only, into
#                     $WORK/score_3settings (no dependency: an early look, not Stop 4's input)
#   index-whole       protein20 k5 whole-protein index rebuilt with --ka-queries 2500
#   search            protein20 k5, 10 chunks, with that fit
#   score             all four settings into $WORK/score, as scripts/342_stop3_submit.sh does
#
# Run from the repository root of a checkout of olgabot/exon-function-342 on a Sherlock login
# node (it only calls sbatch). Job ids go to $WORK/stop3_protein20_jobs.txt.
set -euo pipefail

TABLE=tables/342/342_stop3_resources.tsv
SB=scripts/342_stop3.sbatch
[ -f "$TABLE" ] && [ -f "$SB" ] || { echo "run from the repository root" >&2; exit 2; }
WORK=${NB342_WORK:-$SCRATCH/exon-function-342}
DONE_SETTINGS="aro_hyd_chg_pol4_k11_s1 aro_hyd_arg_chg_pol5_k9_s1 hp_pbotc_1st_ed2_k21_s1"

ask() {  # cpus, memory and time of one row: step alphabet index
    awk -F'\t' -v s="$1" -v a="$2" -v i="$3" '
        !/^#/ && $1 == s && $2 == a && $4 == i {
            printf "--cpus-per-task=%s --mem=%dG --time=%02d:00:00\n", $5, $6, $7; found = 1 }
        END { if (!found) exit 1 }' "$TABLE"
}
submit() {  # name, dependency (or ""), resource args, then sbatch extras; prints the job id
    local name=$1 dep=$2 res=$3; shift 3
    sbatch --parsable --job-name="342-$name" ${dep:+--dependency=afterok:$dep --kill-on-invalid-dep=yes} \
        $res "$@" "$SB"
}

# The three finished settings, each with all 10 chunks, seen through a folder of links so the
# early score cannot pick up protein20's calls while they are being written.
early=$WORK/calls_3settings
mkdir -p "$early"
for s in $DONE_SETTINGS; do
    n=$(ls "$WORK/calls/$s"/chunk_*.calls.tsv.gz 2>/dev/null | wc -l)
    [ "$n" -eq 10 ] || { echo "342: $s has $n of 10 call files" >&2; exit 3; }
    ln -sfn "$WORK/calls/$s" "$early/$s"
done

# The index step skips an index with a DONE file; this one is done but has no fit.
idx=$WORK/indexes/whole.w10.protein20_k5_s1.kmerseek.rocksdb
if [ -f "$idx/DONE" ] && ! ls "$idx"/ka_survival.*.csv > /dev/null 2>&1; then
    rm -f "$idx/DONE"
fi

early_score=$(submit score-3settings "" "$(ask score - -)" \
    --export=ALL,STEP=score,CALLS_DIR=$early,SCORE_OUT=$WORK/score_3settings)
index=$(submit index-whole-protein20 "" "$(ask index protein20 whole)" \
    --export=ALL,STEP=index,ALPHABET=protein20,KSIZE=5,INDEX=whole,KA_QUERIES=2500)
search=$(submit search-protein20 "$index" "$(ask search protein20 regions)" \
    --array=0-9 --export=ALL,STEP=search,ALPHABET=protein20,KSIZE=5)
score=$(submit score "$search" "$(ask score - -)" --export=ALL,STEP=score)

printf 'score-3settings\t%s\nindex-whole-protein20\t%s\nsearch-protein20\t%s\nscore\t%s\n' \
    "$early_score" "$index" "$search" "$score" | tee "$WORK/stop3_protein20_jobs.txt"
