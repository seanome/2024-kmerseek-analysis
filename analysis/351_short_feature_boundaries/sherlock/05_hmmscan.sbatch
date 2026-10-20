#!/bin/bash
#SBATCH --job-name=351-hmmscan
#SBATCH --account=ayeletv
#SBATCH --partition=hns,normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#
# Notebook 351, step 5: one chunk against Pfam-A with hmmscan (HMMER 3.4), default
# thresholds; the E <= 0.01 cutoff on i-Evalue is applied when scoring.
# Sizing: no earlier hmmscan run here; HMMER's memory is about the size of one profile
# and one sequence, so 16 GB is ample. Time is a guess; the ledger records it.
NB351_STEP=hmmscan
source "$NB351_REPO/analysis/351_short_feature_boundaries/sherlock/common.sh"
: "${SLURM_ARRAY_TASK_ID:?}"
CHUNK=$(printf 'chunk%03d' "$SLURM_ARRAY_TASK_ID")
mkdir -p "$D/hmmscan"
ap "$HMMER_IMG" hmmscan --cpu "${SLURM_CPUS_PER_TASK:-16}" --noali \
    --domtblout "$D/hmmscan/$CHUNK.domtbl.partial" "$D/db/Pfam-A.hmm" "$D/chunks/$CHUNK.fa" > /dev/null
mv "$D/hmmscan/$CHUNK.domtbl.partial" "$D/hmmscan/$CHUNK.domtbl"
grep -vc '^#' "$D/hmmscan/$CHUNK.domtbl" || true
