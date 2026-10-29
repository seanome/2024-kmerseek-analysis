#!/bin/bash
# Submit notebook 357's Stop 1 on Sherlock: for each of the four settings, one condensate
# index, then its search (10 chunks) once the index is built, each job with the ask in
# tables/357/357_stop1_resources.tsv. Run from the repository root of a checkout of
# olgabot/nb357-condensate-lead-figure on a Sherlock login node (it only calls sbatch):
#
#   cd /scratch/users/olgabot/2024-kmerseek-analysis-nb357-condensate && bash scripts/357_stop1_submit.sh
#
# A search needs nb 342's whole-protein fit for its setting. If the fit file is missing but
# nb 342's job that makes it (342-index-whole-<alphabet>) is queued or running, the search
# also waits for that job; if neither, the setting is skipped and named. RETRY=1 doubles every
# memory and time ask; finished work is skipped by each step.
set -euo pipefail

TABLE=tables/357/357_stop1_resources.tsv
SB=scripts/357_stop1.sbatch
NB342_WORK=$SCRATCH/exon-function-342
FACTOR=$([ "${RETRY:-0}" = 1 ] && echo 2 || echo 1)
[ -f "$TABLE" ] && [ -f "$SB" ] || { echo "run from the repository root" >&2; exit 2; }
# Sherlock's git is 1.8.3, which has no `git -C`.
[ "$(cd "$NB342_WORK/kmerseek-src" && git rev-parse --short=7 HEAD)" = 9ce5865 ] \
    && [ -x "$NB342_WORK/kmerseek-src/target/release/kmerseek" ] \
    || { echo "357: nb 342's kmerseek build (9ce5865) is not at $NB342_WORK/kmerseek-src" >&2; exit 2; }

ask() {  # cpus, memory and time of one row: step alphabet
    awk -F'\t' -v s="$1" -v a="$2" -v f="$FACTOR" '
        !/^#/ && $1 == s && $2 == a {
            printf "--cpus-per-task=%s --mem=%dG --time=%02d:00:00\n", $4, $5 * f, $6 * f; found = 1 }
        END { if (!found) exit 1 }' "$TABLE"
}
penalty() { awk -F'\t' -v a="$1" '!/^#/ && $1 == a { print $2 }' tables/357/357_penalties.tsv; }
submit() {  # name, dependency (or ""), resource args, then sbatch extras; prints the job id
    local name=$1 dep=$2 res=$3; shift 3
    sbatch --parsable --job-name="357-$name" ${dep:+--dependency=afterok:$dep --kill-on-invalid-dep=yes} $res "$@" "$SB"
}

SETTINGS="aro_hyd_chg_pol4:11 aro_hyd_arg_chg_pol5:9 hp_pbotc_1st_ed2:21 protein20:5"
for s in $SETTINGS; do
    a=${s%%:*}; k=${s##*:}
    fit=$NB342_WORK/indexes/whole.w10.${a}_k${k}_s1.kmerseek.rocksdb/ka_survival.C$(penalty "$a").csv
    fit_job=""
    if [ ! -s "$fit" ]; then
        fit_job=$(squeue --me -h -n "342-index-whole-$a" -o %i | head -1 | sed 's/_.*//')
        [ -n "$fit_job" ] || { echo "$a k$k: SKIPPED, no fit at $fit and no 342-index-whole-$a job queued"; continue; }
    fi
    index=$(submit "index-$a" "" "$(ask index "$a")" --export=ALL,STEP=index,ALPHABET=$a,KSIZE=$k)
    search=$(submit "search-$a" "$index${fit_job:+:$fit_job}" "$(ask search "$a")" \
                 --array=0-9 --export=ALL,STEP=search,ALPHABET=$a,KSIZE=$k)
    echo "$a k$k: index $index, search $search (array 0-9)${fit_job:+, waits for nb 342 fit job $fit_job}"
done
