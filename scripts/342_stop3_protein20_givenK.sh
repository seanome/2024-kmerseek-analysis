#!/bin/bash
# Notebook 342, Stop 3, third pass for protein20 k5 (2026-10-21). Its Karlin-Altschul fit was
# refused in both indexes, also with 2_500 fit queries (job 48685828: 239 million regions,
# but fewer than 4 score bins above the peak hold 30 of each), so its searches stopped with
# exit 4 twice. Here the search is given K instead: the median of the three fitted settings'
# whole-protein K (0.000117; range 0.000085 to 0.000155), with r_database 1, the closed form
# (kmerseek's default with --ka-k). protein20's E-values are therefore not fitted. Its hits
# still come from its own decoy-matched cutoff, set on its own tune-half decoys.
#
# Submits the protein20 search (10 chunks) and, after it (afterok), the four-setting score.
# Run from the repository root of a checkout of olgabot/exon-function-342 on a Sherlock login
# node. Job ids are appended to $WORK/stop3_protein20_jobs.txt.
set -euo pipefail

TABLE=tables/342/342_stop3_resources.tsv
SB=scripts/342_stop3.sbatch
[ -f "$TABLE" ] && [ -f "$SB" ] || { echo "run from the repository root" >&2; exit 2; }
WORK=${NB342_WORK:-$SCRATCH/exon-function-342}
JOBS=$WORK/stop3_protein20_jobs.txt
GIVEN_KA_K=0.000117

queued=$(squeue -h -u "$USER" -n 342-search-protein20,342-score,342-score-3settings -o %i 2>/dev/null || true)
[ -z "$queued" ] || { echo "342: jobs still queued or running: $queued; not submitting" >&2; exit 3; }
[ ! -e "$WORK/calls/protein20_k5_s1" ] || {
    echo "342: $WORK/calls/protein20_k5_s1 exists; check what wrote it before searching again" >&2; exit 3; }

ask() {  # cpus, memory and time of one row: step alphabet index
    awk -F'\t' -v s="$1" -v a="$2" -v i="$3" '
        !/^#/ && $1 == s && $2 == a && $4 == i {
            printf "--cpus-per-task=%s --mem=%dG --time=%02d:00:00\n", $5, $6, $7; found = 1 }
        END { if (!found) exit 1 }' "$TABLE"
}
search=$(sbatch --parsable --job-name=342-search-protein20 $(ask search protein20 regions) --array=0-9 \
    --export=ALL,STEP=search,ALPHABET=protein20,KSIZE=5,GIVEN_KA_K=$GIVEN_KA_K "$SB")
score=$(sbatch --parsable --job-name=342-score $(ask score - -) --dependency=afterok:"$search" \
    --kill-on-invalid-dep=yes --export=ALL,STEP=score "$SB")
printf 'search-protein20 (given K)\t%s\nscore\t%s\n' "$search" "$score" | tee -a "$JOBS"
