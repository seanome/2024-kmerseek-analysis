#!/usr/bin/env bash
# Step 5 comparators on the same chunks and the same held-out references.
#   DNA rule (as readfish does it): minimap2 -x map-ont of each chunk against the family's
#     coding sequences; keep when the chunk has any primary alignment.
#   Slow upper bound: DIAMOND blastx of each chunk against the family's proteins,
#     E <= 1e-3; keep when the chunk has any hit.
# Usage: run_comparators.sh <chunks.fasta> <ref_dir> <holdout> <out_prefix>
set -euo pipefail
E=/Users/olga/anaconda3/envs/adaptive-sampling/bin
chunks=$1; ref=$2; lvl=$3; out=$4
"$E/minimap2" -x map-ont --secondary=no -t 12 "$ref/cds_${lvl}.fasta" "$chunks" 2>/dev/null \
  | awk '$0 ~ /tp:A:P/' > "${out}.minimap2_${lvl}.paf"
"$E/diamond" makedb --in "$ref/ref_${lvl}.fasta" -d "$ref/ref_${lvl}" --quiet
"$E/diamond" blastx -q "$chunks" -d "$ref/ref_${lvl}.dmnd" -o "${out}.diamond_${lvl}.tsv" \
  --evalue 1e-3 -F 15 --sensitive --max-target-seqs 1 --threads 12 --quiet \
  --outfmt 6 qseqid sseqid pident length qstart qend sstart send evalue bitscore
