#!/bin/bash
#SBATCH --job-name=299-search
#SBATCH --account=ayeletv
#SBATCH --partition=hns
#SBATCH --array=0-17
#SBATCH --cpus-per-task=32
#SBATCH --mem=16G
#SBATCH --time=06:00:00
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
# Sizing, measured (array 48667936, 2026-10-20): the 1% gene sample search (about 2.6
# chunks' worth of genes, all nine settings) took 14 s and peaked at 4.7 GB at scaled 100,
# 2.5 GB at scaled 1000; the search against DIAMOND's hit genes, 1 s and 0.15 GB. Searching
# the catalog one chunk at a time keeps memory near 2 GB; 256 chunks at about 6 s each is
# about 30 min. 16 G and 6 h leave room for rows streaming through the reducer (FULL_ROW_CAP
# = 2e9 rows, about 600 GB of text). Scaled 10 sketches hold about 10 times the hashes; its
# gate asks 64 G.
set -euo pipefail

REPO=${SLURM_SUBMIT_DIR:?submit from the repo root}
WORK=${OMRGC_WORK:-$SCRATCH/299-omrgc-pilot}
E=$WORK/env/bin
FULL_ROW_CAP=2000000000
read -r ALPHABET K SCALED < <("$E/python" "$REPO/notebooks/299_omrgc_sketch_pilot.py" setting "$SLURM_ARRAY_TASK_ID")
SID=$ALPHABET.k$K.s$SCALED
OUT=$WORK/search/$SID
mkdir -p "$OUT"
# A setting that already finished (done marker and the gather table) is not searched again,
# so the whole array can be resubmitted after a partial failure.
if [ -e "$OUT/done" ] && [ -s "$OUT/fastmultigather.csv" ]; then
    echo "$SID already done; skipping"
    exit 0
fi
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
    # One manysearch per .zip in the list (299_manysearch_chunks.sh), all streamed into the
    # reducer: memory stays at one chunk's size instead of the whole catalog's.
    local name=$1 queries=$2 against=$3
    : > "$OUT/$name.pairs.tsv"; : > "$OUT/$name.counts.tsv"
    /usr/bin/time -v -o "$OUT/$name.time" bash "$REPO/notebooks/299_manysearch_chunks.sh" \
        "$E/sourmash" "$queries" "$against" "$K" "$ALPHABET" "$SCALED" 32 \
        | awk -v diamond="$OUT/diamond_pairs.tsv" -v counts="$OUT/$name.counts.tsv" -v pairs="$OUT/$name.pairs.tsv" \
            -v bad="$OUT/$name.bad.txt" -f "$REPO/notebooks/299_reduce_manysearch.awk"
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

# BHF regions and their copies through gather, beside the rule, against the 1% gene sample:
# fastmultigather holds every sketch it searches in memory and the whole catalog did not fit
# in 128 GB (array 48667936). A failure here does not lose the results above.
"$E/sourmash" sig grep -i BHF_ "$OUT/queries.sig.zip" -o "$OUT/bhf.sig.zip"
set +e
/usr/bin/time -v -o "$OUT/fastmultigather.time" \
    "$E/sourmash" scripts fastmultigather "$OUT/bhf.sig.zip" "$OUT/sample.txt" "${SEL[@]}" -c 32 -t 0 \
    -o "$OUT/fastmultigather.csv"
echo "fastmultigather exit $?"
set -e
