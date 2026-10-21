#!/bin/bash
#SBATCH --account=ayeletv
#SBATCH --partition=hns
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=301-kmerseek-%x-%j.out
#
# Notebook 301, step 5: one alphabet per job. Walk the alphabet's k values in notebook
# 241's list (notebooks/241_plan.json) from the lowest up; index the step 2 Swiss-Prot
# database at each k until the Karlin-Altschul fit is accepted ("fitted now" in the index
# log); search the step 5 subset at that k only. That k is the alphabet's setting. An
# alphabet with no accepted fit at any k is excluded and listed.
#
# Each chunk's search output goes through a pipe into filter_301_regions.py, which keeps
# the rows with region_evalue <= 10 that touch a scored window; the full CSV is never
# written. A chunk whose filtered file exists is skipped, so a rerun resumes.
#
# Sizing, from the pilot and two memory tests (2026-10-20/21, hp_lehninger2 k19,
# data/301_short_feature_boundaries/pilot/):
#   index: 4-11 min, peak 40-45 GB.
#   search: about 7 s per sequence; 1,450 sequences is about 2.8 h. Memory grows with the
#     number of sequences one process searches (43: 38.5 GB, 200: 68 GB) and does not
#     depend on --batch-size, so the subset is searched in chunks of CHUNK sequences, each
#     a new kmerseek process. 100 per chunk predicts about 50 GB.
#   96 GB covers that and the index; 8 h covers the search plus up to 8 refused k at 11 min.
#
# Needs: ALPHABET, KMERSEEK. Submitted by 301_kmerseek_submit.sh.
set -euo pipefail
: "${SCRATCH:?}" "${SLURM_SUBMIT_DIR:?}" "${ALPHABET:?}" "${KMERSEEK:?}"
REPO=$SLURM_SUBMIT_DIR
BASE=${NB301_DIR:-$SCRATCH/301-short-feature-boundaries}
DB=$BASE/step2/swissprot_minus_query_clusters.fasta
QUERIES=$BASE/step5/subset_queries.fasta
CHUNK=${CHUNK:-100}
OUT=$BASE/step5/$ALPHABET
mkdir -p "$OUT"
module load gcc/14.2.0 python/3.12.1 2>/dev/null || true
TIME=${TIME_BIN:-/usr/bin/time}  # GNU time; a test can point this elsewhere
STATUS=$OUT/status.tsv
printf 'alphabet\tksize\tpenalty\txdrop\tfit\tindex_wall_s\tindex_peak_kb\tsearch_wall_s\tsearch_peak_kb\n' > "$STATUS"

ladder=$(python3 -I -c "
import json
for x in sorted(json.load(open('$REPO/notebooks/241_plan.json')), key=lambda x: x['ksize']):
    if x['alphabet'] == '$ALPHABET':
        print(x['ksize'], x['penalty'], x['xdrop'])
")
# Summed wall time and largest peak RSS over every GNU time block in a file.
wall() { awk -F': ' '/Elapsed/ {n=split($2,t,":"); s=0; for(i=1;i<=n;i++) s=s*60+t[i]; w+=s} END {print w}' "$1"; }
peak() { awk -F': ' '/Maximum resident/ {if ($2>m) m=$2} END {print m}' "$1"; }

while read -r K PENALTY XDROP; do
    TAG=$ALPHABET.k$K.s1.lctrue.c$PENALTY.x$XDROP
    D=$OUT/$TAG
    mkdir -p "$D"
    # A k whose fit was refused on an earlier run keeps its log; it is not indexed again.
    if [ ! -f "$D/idx/CURRENT" ] && ! grep -q "nothing stored" "$D/index.log" 2>/dev/null; then
        "$TIME" -v -o "$D/index.time" "$KMERSEEK" index -i "$DB" -o "$D/idx" \
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
    mkdir -p "$D/chunks"
    awk -v n="$CHUNK" -v d="$D/chunks" '/^>/{if(c%n==0){f=sprintf("%s/chunk%03d.fasta",d,c/n)} c++} {print > f}' "$QUERIES"
    for q in "$D"/chunks/chunk*.fasta; do
        c=$(basename "$q" .fasta)
        [ -s "$D/regions_filtered.$c.tsv.gz" ] && continue
        FIFO=$D/$c.fifo
        rm -f "$FIFO"; mkfifo "$FIFO"
        python3 -I "$REPO/scripts/filter_301_regions.py" --truth "$BASE/step2/truth.tsv" \
            --decoys "$BASE/step3/decoys.tsv" --search-csv "$FIFO" \
            --out "$D/regions_filtered.$c.part.tsv.gz" 2> "$D/filter.$c.log" &
        FILTER=$!
        "$TIME" -v -o "$D/search.$c.time" "$KMERSEEK" search -q "$q" -t "$D/idx" \
            -k "$K" -a "$ALPHABET" --threshold 0 --min-shared-kmers 1 --max-query-pvalue 1 \
            --min-region-score 0 --extend-mismatch-penalty "$PENALTY" --extend-xdrop "$XDROP" \
            -o "$FIFO" > "$D/search.$c.log" 2>&1
        wait "$FILTER"
        rm -f "$FIFO"
        mv "$D/regions_filtered.$c.part.tsv.gz" "$D/regions_filtered.$c.tsv.gz"
    done
    cat "$D"/search.chunk*.time > "$D/search.time"
    printf '%s\t%s\t%s\t%s\tfitted\t%s\t%s\t%s\t%s\n' "$ALPHABET" "$K" "$PENALTY" "$XDROP" \
        "$(wall "$D/index.time")" "$(peak "$D/index.time")" \
        "$(wall "$D/search.time")" "$(peak "$D/search.time")" >> "$STATUS"
    cat "$D"/filter.chunk*.log
    echo "301 step 5: $ALPHABET done at k=$K"
    exit 0
done <<< "$ladder"
echo "301 step 5: $ALPHABET has no accepted Karlin-Altschul fit at any k; excluded"
