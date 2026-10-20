#!/bin/bash
# Rescore the ice-binding run (notebook 295 sections 1-7) after
# scripts/convergence_recover_hmmer_chunks.sh put back the phmmer and jackhmmer chunks
# that had overwritten each other. Same arguments as `make score` (the scoreLists
# process in main.nf), decision rule on, written to a new folder beside results/scored.
# Code is a git archive of commit 4f5ad692 in $C. The last scoring task used 5.4 GB
# for 1 min, so 16 GB and 30 min.
#SBATCH --job-name=score-nb295
#SBATCH --partition=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/users/olgabot/nb295-rescore-4f5ad692/score.%j.log
set -euo pipefail
C=/scratch/users/olgabot/nb295-rescore-4f5ad692
R=/scratch/users/olgabot/2024-kmerseek-convergence-afp/nextflow-runs/convergence-afp
IMG=/scratch/users/olgabot/apptainer-cache/docker.io-olgabot-kmerseek-0.4.0-pr141-3c353d4.img
OUT=$C/out-$SLURM_JOB_ID
mkdir -p $OUT
apptainer exec -B /scratch/users/olgabot $IMG python3 $C/nextflow-runs/convergence-afp/bin/score.py \
  --hits $R/results/hits --truth $C/data/convergence_afp/truth_set.tsv \
  --queries $C/data/convergence_afp/queries.tsv --draws $R/results/inputs/draws.tsv \
  --alphabet-classes $C/nextflow-runs/convergence-afp/assets/alphabet_classes.tsv \
  --afdb-accessions $R/results/db_cache/foldseek/afdb_swissprot/accessions.txt \
  --outdir $OUT/scored 2> $OUT/score.err
echo "done $OUT"
