#!/bin/bash
# Regenerates human_vs_mouse.hp.k21.results.csv.gz, which was truncated ("gzip -t: unexpected
# end of file") -- a stale bad cache from a 2026-03-13 run that predates the 2026-08-10 fix to
# searchHumanVsMouse (kmerseek_human_mouse_orthologs.nf:206-214): a crashed/OOM-killed
# `kmerseek search | gzip` pipe used to exit 0 silently (no pipefail), so storeDir cached the
# truncated file as a permanent "success".
#
# Runs kmerseek directly against Docker (bypassing Nextflow), same pattern as
# run_family_auc_direct.sh / run_rbh_f1_direct.sh: `nextflow run -resume` can't be used here
# because convertResultsToParquet deletes each combo's raw search output from storeDir after
# converting to parquet, and ~99 other combos have already gone through that conversion -- a
# resume would find their raw outputs missing and redo the entire multi-day, ~100-combo sweep
# instead of just this one combo.
#
# The mouse hp k21 rocksdb index itself is ALSO gone from disk (only its .log.gz survived, at
# the top-level results dir, predating the indices/ subdir storeDir reorg) -- this script rebuilds
# it. Note kmerseek's own --help documents `hp` as byte-identical to `hp-lehninger` ("HP Lehninger
# (explicit; identical hashes to hp)"), and a real hp-lehninger k21 index already exists at
# indices/gencode.vM38.pc_translations.canonical.fa.hp_lehninger.k21.scaled1.kmerseek.rocksdb --
# reusing it instead of rebuilding would skip ~12 min. This script rebuilds a standalone `hp`
# index anyway so this combo's provenance matches every other combo (its own index, its own
# search run) rather than quietly borrowing a different combo's cached artifact; swap in
# `--target` pointing at the hp_lehninger dir below and skip step 1 if you'd rather save the time.
#
# Docker Desktop's VM is currently ~98GB (`docker info | grep "Total Memory"`), comfortably above
# the ~6.6GB peak kmerseek itself used in the original (truncated) run's log, so the OOM that
# most likely truncated the original gzip pipe shouldn't recur.
set -euo pipefail

DATA_ROOT=/Users/olga/data/gencode
OUTDIR="$DATA_ROOT/results-human-mouse-orthologs"
INDICES_DIR="$OUTDIR/indices"
ENCODING=hp
KSIZE=21
MIN_SHARED_KMERS=2
MAX_PVALUE=0.05

INDEX_NAME="gencode.vM38.pc_translations.canonical.fa.${ENCODING}.k${KSIZE}.scaled1.kmerseek.rocksdb"
INDEX_DIR_HOST="$INDICES_DIR/$INDEX_NAME"
RESULTS_CSV_HOST="$OUTDIR/human_vs_mouse.${ENCODING}.k${KSIZE}.results.csv.gz"
SEARCH_LOG_HOST="$OUTDIR/human_vs_mouse.${ENCODING}.k${KSIZE}.search.log"
SEARCH_LOG_GZ_HOST="${SEARCH_LOG_HOST}.gz"

mkdir -p "$INDICES_DIR"

echo "=== [1/4] Rebuilding mouse ${ENCODING} k=${KSIZE} index (kmerseek:0.3.1, ~12 min, ~4GB RAM) ==="
INDEX_LOG_TMP=$(mktemp)
docker run --rm --entrypoint "" \
    -v "$DATA_ROOT":/data \
    kmerseek:0.3.1 \
    kmerseek index \
        --input "/data/mouse/m38/gencode.vM38.pc_translations.canonical.fa" \
        --encoding "$ENCODING" \
        --ksize "$KSIZE" \
        --scaled 1 \
        --output "/data/results-human-mouse-orthologs/indices/$INDEX_NAME" \
    2>&1 | tee "$INDEX_LOG_TMP"

if [ ! -d "$INDEX_DIR_HOST" ]; then
    echo "ERROR: expected index dir not found: $INDEX_DIR_HOST" >&2
    exit 1
fi
cp "$INDEX_LOG_TMP" "$INDEX_DIR_HOST/kmerseek_index.log"
echo "Index built: $(du -sh "$INDEX_DIR_HOST" | cut -f1)"

echo "=== [2/4] Searching human vs mouse ${ENCODING} k=${KSIZE} (kmerseek:main, ~28 min, ~7GB RAM) ==="
{
    echo "=== Searching: human vs mouse ${ENCODING} k=${KSIZE} ==="
    echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
} > "$SEARCH_LOG_HOST"

docker run --rm --entrypoint "" \
    -v "$DATA_ROOT":/data \
    kmerseek:main \
    kmerseek search \
        --ksize "$KSIZE" \
        --query "/data/human/v49/gencode.v49.pc_translations.canonical.fa" \
        --target "/data/results-human-mouse-orthologs/indices/$INDEX_NAME" \
        --min-shared-kmers "$MIN_SHARED_KMERS" \
        --max-pvalue "$MAX_PVALUE" \
    2>> "$SEARCH_LOG_HOST" \
    | gzip > "$RESULTS_CSV_HOST"

{
    echo ""
    echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "Compressed size: $(du -sh "$RESULTS_CSV_HOST" | cut -f1)"
} >> "$SEARCH_LOG_HOST"
gzip -f "$SEARCH_LOG_HOST"

echo "=== [3/4] Verifying integrity ==="
gzip -t "$RESULTS_CSV_HOST" && echo "OK: $RESULTS_CSV_HOST passes gzip -t"
gzip -t "$SEARCH_LOG_GZ_HOST" && echo "OK: $SEARCH_LOG_GZ_HOST passes gzip -t"
echo "Row count: $(gzip -dc "$RESULTS_CSV_HOST" | wc -l)"

echo ""
echo "=== [4/4] hp k=21 fixed -- re-running run_rbh_f1_direct.sh to fill the remaining combo ==="
/Users/olga/code/2024-kmerseek-analysis/nextflow-runs/human-mouse-gencode-orthologs/run_rbh_f1_direct.sh
