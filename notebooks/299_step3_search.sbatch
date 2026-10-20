#!/bin/bash
#SBATCH --job-name=299-search
#SBATCH --account=ayeletv
#SBATCH --partition=hns
#SBATCH --array=0-17
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=299-search-%A_%a.out
#
# Notebook 299, step 3b: search all 2_289 queries (109 real, 2_180 shuffled copies), one
# array task per setting. Submitted by 299_gate.sbatch to start when the DIAMOND job
# (step 2) and every 299_step3_sketch_catalog.sbatch task have succeeded.
#
# Search tool: `sourmash scripts manysearch` (branchwater plugin 0.9.15). It reports the
# containment of each query in each gene, the comparison `sourmash search --containment`
# makes, but reads the catalog once for all queries. Hit: at least one shared hash
# (default threshold containment 0.01; every query sketch here has under 100 hashes).
#
# Why three searches. In the laptop test, hp k = 10 at scaled 100 gave 52.7 million hit rows
# for 39_000 genes, because an hp word of 10 letters has only 1_024 possible values. The
# catalog has about 1_200 times as many genes, so such a setting would write tens of
# billions of rows. So:
#   a. recall, exact: real queries against only the genes DIAMOND hit. Whether a query and
#      a gene share a hash does not depend on any other gene.
#   b. hit counts on the 1% random gene sample (seed 299, made by 299_setup.sbatch).
#   c. hit counts on the full catalog, only when (b) predicts at most FULL_ROW_CAP rows.
#      Otherwise the shuffled-hit rate comes from (b) and is marked as estimated.
# manysearch output never touches disk: it streams through 299_reduce_manysearch.awk.
#
# Sizing: guesses until step 1 gives sketch sizes; re-size from `du -sh catalog_sketch/s*`.
# The writer streams at about 70 MB/s (laptop test: 17 GB in about 4 min), about 300 bytes
# per row, so FULL_ROW_CAP = 2e9 rows is about 600 GB and 2.5 h of streaming.
set -euo pipefail

REPO=${SLURM_SUBMIT_DIR:?submit from the repo root}
WORK=${OMRGC_WORK:-$SCRATCH/299-omrgc-pilot}
E=$WORK/env/bin
FULL_ROW_CAP=2000000000
read -r ALPHABET K SCALED < <("$E/python" "$REPO/notebooks/299_omrgc_sketch_pilot.py" setting "$SLURM_ARRAY_TASK_ID")
SID=$ALPHABET.k$K.s$SCALED
OUT=$WORK/search/$SID
mkdir -p "$OUT"
rm -f "$OUT/done"
echo "host $(hostname); $(lscpu | grep 'Model name'); setting $SID"
P="$ALPHABET,k=$K,scaled=$SCALED,noabund"
SEL=(-k "$K" -m "$ALPHABET" -s "$SCALED")

# Query sketches, and how many real queries have no hash at all at this setting.
"$E/sourmash" sketch protein --singleton -p "$P" -o "$OUT/real.sig.zip" "$REPO/notebooks/299_data/queries_real.faa"
"$E/sourmash" sketch protein --singleton -p "$P" -o "$OUT/shuffled.sig.zip" "$REPO/notebooks/299_data/queries_shuffled.faa"
"$E/sourmash" sig cat "$OUT/real.sig.zip" "$OUT/shuffled.sig.zip" -o "$OUT/queries.sig.zip"
"$E/python" - "$OUT" <<'EOF'
import json, sys
from pathlib import Path
import sourmash
out = Path(sys.argv[1])
real = list(sourmash.load_file_as_signatures(str(out / "real.sig.zip")))
empty = [s.name.split()[0] for s in real if len(s.minhash.hashes) == 0]
json.dump(dict(real_queries=len(real), real_queries_with_no_hash=len(empty), names=empty),
          open(out / "query_hashes.json", "w"), indent=1)
print(f"{len(empty)} of {len(real)} real queries have no hash at this setting")
EOF
awk -F'\t' '{print $1"\t"$2"\t"$9}' "$WORK/diamond/hits.tsv" > "$OUT/diamond_pairs.tsv"

# search NAME QUERIES AGAINST_LIST: manysearch streamed through the reducer; writes
# NAME.counts.tsv, NAME.pairs.tsv, NAME.bad.txt and NAME.time.
search() {
    local name=$1 queries=$2 against=$3
    rm -f "$OUT/$name.fifo"; mkfifo "$OUT/$name.fifo"
    : > "$OUT/$name.pairs.tsv"; : > "$OUT/$name.counts.tsv"
    awk -v diamond="$OUT/diamond_pairs.tsv" -v counts="$OUT/$name.counts.tsv" -v pairs="$OUT/$name.pairs.tsv" \
        -v bad="$OUT/$name.bad.txt" -f "$REPO/notebooks/299_reduce_manysearch.awk" < "$OUT/$name.fifo" &
    local reducer=$!
    /usr/bin/time -v -o "$OUT/$name.time" \
        "$E/sourmash" scripts manysearch "$queries" "$against" "${SEL[@]}" -c 32 --ignore-abundance -N \
        -o "$OUT/$name.fifo"
    wait "$reducer"
    rm -f "$OUT/$name.fifo"
    echo "$name: $(awk -F'\t' '{s += $2} END {print s + 0}' "$OUT/$name.counts.tsv") rows, bad lines $(cat "$OUT/$name.bad.txt")"
}

# a. Recall: sketch the genes DIAMOND hit, 32 parts in parallel, then search real queries.
mkdir -p "$OUT/hit_genes"
ls "$WORK/diamond/hit_gene_parts"/*.fna* | xargs -P 32 -I{} sh -c \
    '"$0" sketch translate --singleton -p "$1" -o "$2/hit_genes/$(basename {}).sig.gz" {} &&
     "$3" "$4" drop-empty "$2/hit_genes/$(basename {}).sig.gz" "$2/hit_genes/$(basename {}).sig.zip"' \
    "$E/sourmash" "$P" "$OUT" "$E/python" "$REPO/notebooks/299_omrgc_sketch_pilot.py"
ls "$OUT/hit_genes"/*.sig.zip > "$OUT/hit_genes.txt"
search recall "$OUT/real.sig.zip" "$OUT/hit_genes.txt"

# b. Hit counts on the 1% random gene sample.
ls "$WORK/catalog_sketch/s$SCALED/sample"/part*.sig.zip > "$OUT/sample.txt"
search sample "$OUT/queries.sig.zip" "$OUT/sample.txt"

# c. Hit counts on the full catalog when the sample predicts few enough rows.
SAMPLE_ROWS=$(awk -F'\t' '{s += $2} END {print s + 0}' "$OUT/sample.counts.tsv")
PREDICTED=$("$E/python" -c "import json; s = json.load(open('$WORK/catalog_stats.json')); print(int($SAMPLE_ROWS * s['catalog_genes'] / s['sample_genes']))")
echo "predicted full-catalog rows: $PREDICTED (cap $FULL_ROW_CAP)"
if [ "$PREDICTED" -le "$FULL_ROW_CAP" ]; then
    ls "$WORK/catalog_sketch/s$SCALED"/chunk*.sig.zip > "$OUT/catalog.txt"
    search full "$OUT/queries.sig.zip" "$OUT/catalog.txt"
    COUNT_SOURCE=full
else
    COUNT_SOURCE=sample_1pct
fi
printf '{"count_source": "%s", "sample_rows": %s, "predicted_full_rows": %s, "full_row_cap": %s}\n' \
    "$COUNT_SOURCE" "$SAMPLE_ROWS" "$PREDICTED" "$FULL_ROW_CAP" > "$OUT/status.json"
touch "$OUT/done"

# BHF regions and their copies through gather, beside the rule. A failure here (memory, for
# example) does not lose the results above; it is logged and reported as not done.
"$E/sourmash" sig grep -i BHF_ "$OUT/queries.sig.zip" -o "$OUT/bhf.sig.zip"
ls "$WORK/catalog_sketch/s$SCALED"/chunk*.sig.zip > "$OUT/catalog.txt"
set +e
/usr/bin/time -v -o "$OUT/fastmultigather.time" \
    "$E/sourmash" scripts fastmultigather "$OUT/bhf.sig.zip" "$OUT/catalog.txt" "${SEL[@]}" -c 32 -t 0 \
    -o "$OUT/fastmultigather.csv"
echo "fastmultigather exit $?"
set -e
