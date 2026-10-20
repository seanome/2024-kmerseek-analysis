#!/bin/bash
#SBATCH --job-name=351-blastp
#SBATCH --account=ayeletv
#SBATCH --partition=hns,normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#
# Notebook 351, step 6: one chunk against the excluded Swiss-Prot with blastp (BLAST+
# 2.17.0), E <= 0.01, default 500 targets per query. Every HSP is written; the scoring
# keeps the best-scoring HSP of each target.
# Sizing: no earlier blastp run here; Swiss-Prot's BLAST database is under 1 GB.
NB351_STEP=blastp
source "$NB351_REPO/analysis/351_short_feature_boundaries/sherlock/common.sh"
: "${SLURM_ARRAY_TASK_ID:?}"
CHUNK=$(printf 'chunk%03d' "$SLURM_ARRAY_TASK_ID")
mkdir -p "$D/blastp"
ap "$BLAST_IMG" blastp -query "$D/chunks/$CHUNK.fa" -db "$D/db/swissprot_excluded" \
    -evalue "$EVALUE" -num_threads "${SLURM_CPUS_PER_TASK:-16}" \
    -outfmt "6 qseqid sseqid pident length qstart qend sstart send evalue bitscore qseq sseq" \
    -out "$D/blastp/$CHUNK.tsv.partial"
mv "$D/blastp/$CHUNK.tsv.partial" "$D/blastp/$CHUNK.tsv"
wc -l < "$D/blastp/$CHUNK.tsv"
