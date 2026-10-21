#!/bin/bash
#SBATCH --account=ayeletv
#SBATCH --partition=hns
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=301-kmerseek-%x-%j.out
#
# Notebook 301, step 5: one alphabet per job. Walk the alphabet's k values in notebook
# 241's list (notebooks/241_plan.json) from the lowest up; index the step 2 Swiss-Prot
# database at each k until the Karlin-Altschul fit is accepted ("fitted now" in the index
# log); search the step 5 subset at that k only. That k is the alphabet's setting. An
# alphabet with no accepted fit at any k is excluded and listed.
#
# The search output goes through a pipe into filter_301_regions.py, which keeps the rows
# with region_evalue <= 10 that touch a scored window; the full CSV is never written.
#
# Sizing, from the pilot (2026-10-20, data/301_short_feature_boundaries/pilot/):
#   index: 4-11 min, peak 40-45 GB.
#   search: about 6.5 s per sequence; 1,450 sequences is about 2.6 h. Memory is about
#     3 GB plus 0.85 GB per query in a batch; --batch-size 16 predicts about 17 GB.
#   64 GB covers the index; 8 h covers the search plus up to 8 refused k values at 11 min.
#
# Needs: ALPHABET, KMERSEEK. Submitted by 301_kmerseek_submit.sh.
set -euo pipefail
: "${SCRATCH:?}" "${SLURM_SUBMIT_DIR:?}" "${ALPHABET:?}" "${KMERSEEK:?}"
REPO=$SLURM_SUBMIT_DIR
BASE=${NB301_DIR:-$SCRATCH/301-short-feature-boundaries}
DB=$BASE/step2/swissprot_minus_query_clusters.fasta
QUERIES=$BASE/step5/subset_queries.fasta
OUT=$BASE/step5/$ALPHABET
mkdir -p "$OUT"
module load gcc/14.2.0 python/3.12.1
STATUS=$OUT/status.tsv
[ -s "$STATUS" ] || printf 'alphabet\tksize\tpenalty\txdrop\tfit\tindex_wall_s\tindex_peak_kb\tsearch_wall_s\tsearch_peak_kb\n' > "$STATUS"

ladder=$(python3 -I -c "
import json
for x in sorted(json.load(open('$REPO/notebooks/241_plan.json')), key=lambda x: x['ksize']):
    if x['alphabet'] == '$ALPHABET':
        print(x['ksize'], x['penalty'], x['xdrop'])
")
wall() { awk -F': ' '/Elapsed/ {n=split($2,t,":"); s=0; for(i=1;i<=n;i++) s=s*60+t[i]; print s}' "$1"; }
peak() { awk -F': ' '/Maximum resident/ {print $2}' "$1"; }

while read -r K PENALTY XDROP; do
    TAG=$ALPHABET.k$K.s1.lctrue.c$PENALTY.x$XDROP
    D=$OUT/$TAG
    mkdir -p "$D"
    if [ ! -f "$D/idx/CURRENT" ]; then
        /usr/bin/time -v -o "$D/index.time" "$KMERSEEK" index -i "$DB" -o "$D/idx" \
            -k "$K" -s 1 -a "$ALPHABET" --remove-low-complexity \
            --extend-mismatch-penalty "$PENALTY" --extend-xdrop "$XDROP" \
            --ka-queries 200 --ka-survival-out "$D/ka_survival.csv" > "$D/index.log" 2>&1
    fi
    if ! grep -q "fitted now" "$D/index.log"; then
        printf '%s\t%s\t%s\t%s\trefused\t%s\t%s\t\t\n' "$ALPHABET" "$K" "$PENALTY" "$XDROP" \
            "$(wall "$D/index.time")" "$(peak "$D/index.time")" >> "$STATUS"
        rm -rf "$D/idx"   # 4-5 GB each; the log and fit table stay
        continue
    fi
    FIFO=$D/search.fifo
    rm -f "$FIFO"; mkfifo "$FIFO"
    python3 -I "$REPO/scripts/filter_301_regions.py" --truth "$BASE/step2/truth.tsv" \
        --decoys "$BASE/step3/decoys.tsv" --search-csv "$FIFO" \
        --out "$D/regions_filtered.tsv.gz" 2> "$D/filter.log" &
    FILTER=$!
    /usr/bin/time -v -o "$D/search.time" "$KMERSEEK" search -q "$QUERIES" -t "$D/idx" \
        -k "$K" -a "$ALPHABET" --threshold 0 --min-shared-kmers 1 --max-query-pvalue 1 \
        --min-region-score 0 --extend-mismatch-penalty "$PENALTY" --extend-xdrop "$XDROP" \
        --batch-size 16 -o "$FIFO" > "$D/search.log" 2>&1
    wait "$FILTER"
    rm -f "$FIFO"
    printf '%s\t%s\t%s\t%s\tfitted\t%s\t%s\t%s\t%s\n' "$ALPHABET" "$K" "$PENALTY" "$XDROP" \
        "$(wall "$D/index.time")" "$(peak "$D/index.time")" \
        "$(wall "$D/search.time")" "$(peak "$D/search.time")" >> "$STATUS"
    cat "$D/filter.log"
    echo "301 step 5: $ALPHABET done at k=$K"
    exit 0
done <<< "$ladder"
echo "301 step 5: $ALPHABET has no accepted Karlin-Altschul fit at any k; excluded"
