#!/usr/bin/env bash
# Step 2 truth: DIAMOND blastx of every full-length read against all of Swiss-Prot 2026_03,
# E <= 1e-5, best hit per read. Frameshift-aware alignment (-F 15) because nanopore reads
# carry insertions and deletions.
# Usage: run_truth_diamond.sh <reads.fastq> <sprot.dmnd> <out.tsv>
set -euo pipefail
DIAMOND=/Users/olga/anaconda3/envs/adaptive-sampling/bin/diamond
"$DIAMOND" blastx -q "$1" -d "$2" -o "$3" --evalue 1e-5 -F 15 --sensitive \
  --max-target-seqs 1 --threads 12 --quiet \
  --outfmt 6 qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore qlen slen
