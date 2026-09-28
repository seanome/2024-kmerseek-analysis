#!/bin/zsh
# run_diamond_fresh.sh
#
# Fresh DIAMOND --ultra-sensitive (E<=10) human-vs-species search for all 9
# QfO species, used by construct_pair_strata.py to classify positive pairs
# into 'hard' (no HSP found) / 'easy' (HSP found) strata.
#
# Why this exists instead of reusing nextflow-runs/pfam-benchmark-tools'
# diamondSearch output: that output is broken — verified empty (20-byte gzip
# stub, zero HSPs) for 6/9 species and missing entirely for 2 more (worm,
# arabidopsis), a container-execution bug in that older containerized run.
# The exact same diamond command run directly (this script) works fine.
#
# Idempotent: skips any species whose output file already exists.
#
# Usage:
#   scripts/run_diamond_fresh.sh

set -o pipefail
set -e

QFO=${QFO:-$HOME/data/quest-for-orthologs/QfO_release_2020_04_with_updated_UP000008143}
DIAMOND=${DIAMOND:-/Users/olga/anaconda3/envs/orthofinder/bin/diamond}
CACHE=${CACHE:-$(cd "$(dirname "$0")/.." && pwd)/results/pfam_benchmark/diamond_fresh}
HUMAN_FASTA="$QFO/Eukaryota/UP000005640_9606.fasta"

mkdir -p "$CACHE"

species_list=(mouse chicken zebrafish ciona fly worm yeast arabidopsis ecoli)

fasta_for() {
  case "$1" in
    mouse) echo "Eukaryota/UP000000589_10090.fasta" ;;
    chicken) echo "Eukaryota/UP000000539_9031.fasta" ;;
    zebrafish) echo "Eukaryota/UP000000437_7955.fasta" ;;
    ciona) echo "Eukaryota/UP000008144_7719.fasta" ;;
    fly) echo "Eukaryota/UP000000803_7227.fasta" ;;
    worm) echo "Eukaryota/UP000001940_6239.fasta" ;;
    yeast) echo "Eukaryota/UP000002311_559292.fasta" ;;
    arabidopsis) echo "Eukaryota/UP000006548_3702.fasta" ;;
    ecoli) echo "Bacteria/UP000000625_83333.fasta" ;;
  esac
}

for sp in $species_list; do
  out="$CACHE/human_vs_${sp}.diamond.tsv.gz"
  if [ -f "$out" ]; then
    echo "$sp: already done, skipping"
    continue
  fi
  echo "=== $sp ==="
  rel=$(fasta_for "$sp")
  db="$CACHE/${sp}_db"
  "$DIAMOND" makedb --in "$QFO/$rel" --db "$db" --threads 8 --quiet
  "$DIAMOND" blastp \
    --query "$HUMAN_FASTA" \
    --db "${db}.dmnd" \
    --ultra-sensitive \
    --outfmt 6 qseqid sseqid bitscore evalue \
    --evalue 10.0 \
    --max-target-seqs 0 \
    --threads 8 \
    --block-size 2 \
    --out "$CACHE/human_vs_${sp}.tsv" --quiet
  gzip -c "$CACHE/human_vs_${sp}.tsv" > "$out"
  rm -f "$CACHE/human_vs_${sp}.tsv" "${db}.dmnd"
  n=$(gzip -dc "$out" | wc -l)
  echo "$sp: $n HSPs"
done
echo "ALL DONE"
