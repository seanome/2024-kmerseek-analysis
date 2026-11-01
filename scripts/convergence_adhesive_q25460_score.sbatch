#!/bin/bash
#SBATCH --job-name=score-q25460
#SBATCH --partition=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/users/olgabot/q25460-score-acbd1af4/score.%j.log
set -euo pipefail
C=/scratch/users/olgabot/q25460-score-acbd1af4
R=/scratch/users/olgabot/2024-kmerseek-convergence-afp/nextflow-runs/convergence-afp
IMG=/scratch/users/olgabot/apptainer-cache/docker.io-olgabot-kmerseek-0.4.0-pr141-3c353d4.img
OUT=$C/out-$(date +%Y%m%dT%H%M)
mkdir -p $OUT
ls $R/results-adhesive-q25460/hits/kmerseek/*.status.tsv | wc -l > $OUT/n_status_files.txt
run() { apptainer exec -B /scratch/users/olgabot $IMG python3 "$@"; }
run $C/nextflow-runs/convergence-afp/bin/score.py --no-decide \
  --hits $R/results-adhesive-q25460/hits --truth $C/data/convergence_adhesive/truth_set.tsv \
  --queries $C/data/convergence_adhesive/queries.tsv --draws $R/results-adhesive-q25460/inputs/draws.tsv \
  --alphabet-classes $C/nextflow-runs/convergence-afp/assets/alphabet_classes.tsv \
  --afdb-accessions $R/results/db_cache/foldseek/afdb_swissprot/accessions.txt \
  --outdir $OUT/scored 2> $OUT/score.err
run $C/scripts/convergence_adhesive_q25460_detail.py \
  --hits $R/results-adhesive-q25460/hits --draws $R/results-adhesive-q25460/inputs/draws.tsv \
  --truth $C/data/convergence_adhesive/truth_set.tsv --swissprot $R/data/uniprot_sprot.2026_03.fasta.gz \
  --outdir $OUT/detail
echo "done $OUT"
