#!/bin/bash
# Scores slurm/kamrat_reads.sbatch against the paper's lists. Writes $S/kamrat_reads_score.
#SBATCH --job-name=nb361-kreads-score
#SBATCH --partition=normal,hns
#SBATCH --account=ayeletv
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=/scratch/users/olgabot/nb361-pec/logs_kamrat_reads/%x.%j.out
set -euo pipefail
S=/scratch/users/olgabot/nb361-pec
R=/scratch/users/olgabot/2024-kmerseek-analysis-nb361/analysis/pec-domen2026-splash
$S/env/bin/python $R/scripts/kamrat_reads_score.py $S $R/design/label_swap_kamrat/manifest.tsv $S/kamrat_reads_score
sacct -j ${JOB_ID:?} --format=JobID%16,State,Elapsed,MaxRSS -P > $S/kamrat_reads_score/resources.txt
