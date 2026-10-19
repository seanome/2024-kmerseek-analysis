#!/bin/bash
#SBATCH --job-name=299-check
#SBATCH --account=ayeletv
#SBATCH --partition=hns
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=299-check-%j.out
#
# Notebook 299: why did OM-RGC_v2_genes.fna.gz fail its gzip check (setup job 48522928,
# "crc32 mismatch") although its size is right and 64 sampled 1 MB windows, including every
# range boundary, match fresh fetches from EBI? Three checks:
#   1. a second full download, 32 parallel ranges, compared byte for byte with the first;
#      equal files mean EBI serves this content, so the source file itself is damaged;
#   2. full decompression of the first copy: bytes, genes, the last line, and gzip's own
#      error message (where the stream breaks);
#   3. gzip check of OM-RGC_v2.tsv.gz (10.3 GB, the same catalog as a table with a
#      nucleotide sequence column), the fallback source.
# Sizing: downloads measured at 29.6 GB in about 30 min (job 48522928); gzip -dc of ~30 GB
# about 15-20 min on one core; 8 G covers the pipes. 4 h is a wide margin.
set -uo pipefail

WORK=${OMRGC_WORK:-$SCRATCH/299-omrgc-pilot}
E=$WORK/env/bin
BASE=https://ftp.ebi.ac.uk/biostudies/fire/S-BSST/S-BSSTxxx297/S-BSST297/Files
GZ=$WORK/catalog/OM-RGC_v2_genes.fna.gz
CHECK=$WORK/check
mkdir -p "$CHECK"
echo "host $(hostname)"

fetch() {  # fetch URL BYTES OUT: 32 parallel byte ranges, each resuming on a stall
    local url=$1 total=$2 out=$3 n=32 step i
    step=$(( (total + n - 1) / n ))
    for i in $(seq 0 $(( n - 1 ))); do
        (
            first=$(( i * step )); last=$(( first + step - 1 ))
            (( last >= total )) && last=$(( total - 1 ))
            want=$(( last - first + 1 ))
            for attempt in $(seq 1 50); do
                have=$(stat -c %s "$out.$i" 2>/dev/null || echo 0)
                [ "$have" -ge "$want" ] && break
                curl -sfL --speed-limit 100000 --speed-time 60 -r "$(( first + have ))-$last" "$url" >> "$out.$i"
            done
        ) &
    done
    wait
    for i in $(seq 0 $(( n - 1 ))); do cat "$out.$i"; done > "$out"
    rm -f "$out".[0-9]*
    echo "$out: $(stat -c %s "$out") bytes (expected $total)"
}

# 3 first (smallest), in the background while 1 and 2 run.
(
    fetch "$BASE/OM-RGC_v2.tsv.gz" 10320406416 "$CHECK/OM-RGC_v2.tsv.gz"
    gzip -t "$CHECK/OM-RGC_v2.tsv.gz" 2> "$CHECK/tsv_gzip_test.txt"
    echo "tsv gzip -t exit $?: $(cat "$CHECK/tsv_gzip_test.txt")"
) &
TSV=$!

# 2. Full decompression of the first copy: bytes, genes and last line from one awk pass.
(
    gzip -dc "$GZ" 2> "$CHECK/fna_gzip_error.txt" \
        | awk '/^>/ {n++} {b += length($0) + 1; last = $0} END {print b, n; print substr(last, 1, 150)}' \
        > "$CHECK/fna_summary.txt"
    echo "fna decompressed (bytes, genes, last line): $(cat "$CHECK/fna_summary.txt" | tr '\n' ' ')"
    echo "gzip said: $(cat "$CHECK/fna_gzip_error.txt")"
) &
DEC=$!

# 1. Second full download, then compare.
fetch "$BASE/OM-RGC_v2_genes.fna.gz" 29607178264 "$CHECK/OM-RGC_v2_genes.fna.gz.second"
if cmp "$GZ" "$CHECK/OM-RGC_v2_genes.fna.gz.second" > "$CHECK/cmp.txt" 2>&1; then
    echo "second download identical to the first: EBI serves this content"
else
    echo "second download differs: $(cat "$CHECK/cmp.txt")"
fi
md5sum "$GZ" "$CHECK/OM-RGC_v2_genes.fna.gz.second"
wait "$DEC" "$TSV"
