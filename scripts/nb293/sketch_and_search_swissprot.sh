#!/bin/bash
# Step 6 of notebook 293: sourmash sketches of every Swiss-Prot protein in the chosen
# alphabet and k (scaled = 1, every k-mer kept), a RocksDB index, and the search of every
# protein against the index, keeping pairs where the query's containment in the match is
# at least THRESHOLD. Usage: sketch_and_search_swissprot.sh OUT_DIR SWISSPROT_FASTA_GZ MOLTYPE K THRESHOLD
set -euo pipefail
OUT=$1; SP=$2; MOLTYPE=$3; K=$4; T=$5
S=/Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/sourmash
mkdir -p "$OUT"
printf 'name,genome_filename,protein_filename\nswissprot,,%s\n' "$SP" > "$OUT/manysketch.csv"
/usr/bin/time -l "$S" scripts manysketch "$OUT/manysketch.csv" -p "$MOLTYPE,k=$K,scaled=1" --singleton -o "$OUT/swissprot.$MOLTYPE.k$K.sig.zip" 2> "$OUT/manysketch.time"
/usr/bin/time -l "$S" scripts index "$OUT/swissprot.$MOLTYPE.k$K.sig.zip" -m "$MOLTYPE" -k "$K" -s 1 -o "$OUT/swissprot.$MOLTYPE.k$K.rocksdb" 2> "$OUT/index.time"
/usr/bin/time -l "$S" scripts manysearch "$OUT/swissprot.$MOLTYPE.k$K.sig.zip" "$OUT/swissprot.$MOLTYPE.k$K.rocksdb" -m "$MOLTYPE" -k "$K" -s 1 -t "$T" --ignore-abundance -N -o "$OUT/manysearch.$MOLTYPE.k$K.t$T.csv" 2> "$OUT/manysearch.time"
tail -n 20 "$OUT"/*.time | grep -E "==>|real|maximum resident"
