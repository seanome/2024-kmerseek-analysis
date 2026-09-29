#!/usr/bin/env bash
# Notebook 254, build check: rerun the midi-plus image on one case (MYB against arabidopsis,
# hp_kyte_doolittle2 k19, mask on) with the midi-plus index and search options, to see
# where the old build ends the exact run. Writes old_on.csv under $OUT.
set -euo pipefail
OUT=/Users/olga/data/hero-244-extension/oldcheck
QFO=/Users/olga/data/quest-for-orthologs/QfO_release_2020_04_with_updated_UP000008143
IMG=olgabot/kmerseek:2026-08-24-reduced-alphabets
mkdir -p "$OUT"
awk '/^>/{k=($0 ~ /\|P10242\|/)} k' "$QFO/Eukaryota/UP000005640_9606.fasta" > "$OUT/q.fa"
/bin/rm -rf "$OUT/idx"
docker run --rm --platform linux/amd64 -v "$OUT:$OUT" -v "$QFO:$QFO" "$IMG" kmerseek index \
    --alphabet hp_kyte_doolittle2 --ksize 19 --input "$QFO/Eukaryota/UP000006548_3702.fasta" \
    --output "$OUT/idx" --remove-low-complexity > "$OUT/idx.log" 2>&1
docker run --rm --platform linux/amd64 -v "$OUT:$OUT" "$IMG" kmerseek search \
    --alphabet hp_kyte_doolittle2 --ksize 19 --query "$OUT/q.fa" --target "$OUT/idx" \
    --remove-low-complexity --threshold 0 --min-shared-kmers 2 --max-query-pvalue 0.05 \
    --min-region-score 1.3 > "$OUT/old_on.csv" 2> "$OUT/search.log"
