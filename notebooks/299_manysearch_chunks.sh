#!/bin/bash
# Notebook 299: run `sourmash scripts manysearch` once per catalog chunk and write every
# chunk's CSV to stdout, for 299_reduce_manysearch.awk to read as one stream.
#
#   299_manysearch_chunks.sh SOURMASH QUERIES CHUNK_LIST KSIZE MOLTYPE SCALED CPUS
#
# Why per chunk: manysearch keeps the whole collection it searches against in memory. The
# whole catalog (256 chunk .zip files, all nine settings) needed over 128 GB (search array
# 48667936, every task out of memory), while the 1% gene sample, about 2.6 chunks' worth,
# peaked at 4.7 GB at scaled 100. One chunk at a time stays near 2 GB. A query's hits in the
# whole catalog are the union of its hits in each chunk, so the reducer's counts are the
# same (checked on 39_000 genes split in two: identical per-query counts and pairs).
set -euo pipefail
SOURMASH=$1 QUERIES=$2 LIST=$3 K=$4 MOLTYPE=$5 SCALED=$6 CPUS=$7
while read -r CHUNK; do
    "$SOURMASH" scripts manysearch "$QUERIES" "$CHUNK" -k "$K" -m "$MOLTYPE" -s "$SCALED" \
        -c "$CPUS" --ignore-abundance -N -o /dev/stdout 2> /dev/null
done < "$LIST"
