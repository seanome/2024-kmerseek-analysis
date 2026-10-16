#!/bin/bash
#SBATCH --job-name=qs-identity
#SBATCH --account=ayeletv
#SBATCH --partition=hns,normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/users/olgabot/leaderboard-query-strata/logs/%x-%j.log
#
# Three query-side strata that need only the query FASTA (and a target proteome):
#   1. identity to the nearest protein in each of the 15 target proteomes (MMseqs2 -s 7.0)
#   3. low complexity: residues SEG masks, with SEG's defaults (window 12, 2.2, 2.5)
#   4. protein length
# Ask: 8 CPUs, 16 GB, 2 h. Measured history: mmseqs2Search in the midi-plus run searched the
# same 998 queries against these proteomes at peak 1.8 GB on 16 CPUs (nextflow.config
# comment and trace run-midi-plus/qfo_pfam_region.2026-09-01.trace.txt); -s 7.0 is slower
# than that run's setting, so the time is padded.
set -euo pipefail
source /scratch/users/olgabot/leaderboard-query-strata/repo/scripts/query_strata/paths.sh
OUT=$WD/out/identity
mkdir -p "$OUT" "$WD/out/seg" "$WD/out/length"
TMP=${L_SCRATCH:-$WD/tmp}/mmseqs-tmp
mkdir -p "$TMP"
B="--bind $WD --bind $QFO_PIPE"

apptainer exec $B "$MMSEQS_IMG" mmseqs version | tee "$OUT/mmseqs_version.txt"

# 1. One search per target. Default output columns: query, target, fident, alnlen,
# mismatch, gapopen, qstart, qend, tstart, tend, evalue, bits. -e 1e-3 keeps only hits the
# bins can use; "no hit" means nothing at E < 1e-3.
while read -r label rel; do
  [ -s "$OUT/$label.m8" ] && { echo "have $label"; continue; }
  apptainer exec $B "$MMSEQS_IMG" mmseqs easy-search \
    "$QUERY_FASTA" "$QFO_DIR/$rel.fasta" "$OUT/$label.m8.part" "$TMP/$label" \
    -s 7.0 -e 1e-3 --threads "$SLURM_CPUS_PER_TASK"
  mv "$OUT/$label.m8.part" "$OUT/$label.m8"
done <<< "$TARGETS"

# Best hit per query: lowest E-value, then highest bit score, then target name, so a tie
# always resolves the same way.
for f in "$OUT"/*.m8; do
  label=$(basename "$f" .m8)
  sort -t $'\t' -k1,1 -k11,11g -k12,12gr -k2,2 "$f" \
    | awk -F'\t' -v OFS='\t' -v sp="$label" '$1 != prev {print sp, $0; prev = $1}'
done > "$OUT/best_hits.tsv"
wc -l "$OUT/best_hits.tsv"

# 3. SEG through NCBI segmasker, which implements SEG with SEG's own defaults
# (-window 12 -locut 2.2 -hicut 2.5). Interval output: 0-based inclusive start - end.
apptainer exec $B "$BLAST_IMG" segmasker -version | tee "$WD/out/seg/segmasker_version.txt"
apptainer exec $B "$BLAST_IMG" segmasker -in "$QUERY_FASTA" -infmt fasta \
  -outfmt interval -out "$WD/out/seg/queries.seg.intervals"

# 4. Length: residues per sequence.
awk '/^>/{if(id)print id"\t"n; id=substr($1,2); n=0; next}{n+=length($0)}END{print id"\t"n}' \
  "$QUERY_FASTA" > "$WD/out/length/queries.length.tsv"
wc -l "$WD/out/length/queries.length.tsv"

sha256sum "$OUT"/*.m8 "$OUT/best_hits.tsv" "$WD/out/seg/queries.seg.intervals" \
  "$WD/out/length/queries.length.tsv" "$QUERY_FASTA" > "$WD/out/identity_seg_length.sha256"
