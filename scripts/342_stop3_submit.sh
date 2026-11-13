#!/bin/bash
# Submit notebook 342's Stop 3 on Sherlock: build -> decoys -> 8 indexes -> 4 searches (10
# chunks each) -> scoring, each job with the ask in tables/342/342_stop3_resources.tsv and an
# afterok dependency on the jobs it reads. Run from the repository root of a checkout of
# olgabot/exon-function-342 on a Sherlock login node (it only calls sbatch):
#
#   cd $SCRATCH/2024-kmerseek-analysis-342 && bash scripts/342_stop3_submit.sh
#
# RETRY=1 doubles every memory and time ask (for a resubmission after a kill); finished work
# is skipped by each step, so a resubmission redoes only what is missing.
set -euo pipefail

TABLE=tables/342/342_stop3_resources.tsv
SB=scripts/342_stop3.sbatch
FACTOR=$([ "${RETRY:-0}" = 1 ] && echo 2 || echo 1)
[ -f "$TABLE" ] && [ -f "$SB" ] || { echo "run from the repository root" >&2; exit 2; }

ask() {  # cpus, memory and time of one row: step alphabet index
    awk -F'\t' -v s="$1" -v a="$2" -v i="$3" -v f="$FACTOR" '
        !/^#/ && $1 == s && $2 == a && $4 == i {
            printf "--cpus-per-task=%s --mem=%dG --time=%02d:00:00\n", $5, $6 * f, $7 * f; found = 1 }
        END { if (!found) exit 1 }' "$TABLE"
}
submit() {  # name, dependency (or ""), resource args, then sbatch extras; prints the job id
    local name=$1 dep=$2 res=$3; shift 3
    sbatch --parsable --job-name="342-$name" ${dep:+--dependency=afterok:$dep} $res "$@" "$SB"
}

build=$(submit build "" "$(ask build - -)" --export=ALL,STEP=build)
decoys=$(submit decoys "" "$(ask decoys - -)" --export=ALL,STEP=decoys)
echo "build $build, decoys $decoys"

SETTINGS="aro_hyd_chg_pol4:11 aro_hyd_arg_chg_pol5:9 hp_pbotc_1st_ed2:21 protein20:5"
searches=()
for s in $SETTINGS; do
    a=${s%%:*}; k=${s##*:}
    ids=()
    for index in regions whole; do
        ids+=("$(submit "index-$index-$a" "$build:$decoys" "$(ask index "$a" "$index")" \
                 --export=ALL,STEP=index,ALPHABET=$a,KSIZE=$k,INDEX=$index)")
    done
    search=$(submit "search-$a" "${ids[0]}:${ids[1]}" "$(ask search "$a" regions)" \
                 --array=0-9 --export=ALL,STEP=search,ALPHABET=$a,KSIZE=$k)
    searches+=("$search")
    echo "$a k$k: regions index ${ids[0]}, whole index ${ids[1]}, search $search (array 0-9)"
done
dep=$(IFS=:; echo "${searches[*]}")
score=$(submit score "$dep" "$(ask score - -)" --export=ALL,STEP=score)
echo "score $score"
