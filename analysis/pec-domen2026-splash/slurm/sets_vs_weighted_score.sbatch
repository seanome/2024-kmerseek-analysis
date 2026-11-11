#!/bin/bash
# Scores version A (weighted) and version B (sets only) for every Botryllus set, kmerseek setting and
# human comparison (design/sets_vs_weighted/decision_rules.md, scripts/sets_vs_weighted_score.py).
# Run after sets_vs_weighted_human_counts.sbatch and sets_vs_weighted_human_sets.sbatch (afterok).
# Writes $W/results/.
# Resources: version A measured on PR 201's counts (job 51944335, 2026-11-11): 3 Botryllus sets, 3
# settings and 10 human comparisons took 48 s and 0.66 GB, and gave PR 201's D in all 27 rows. Here
# 10 sets, 11 settings and 14 comparisons, about 11 times the work: under 10 min and about 6 GB.
# Version B is scaled, not measured: it reads up to 638 human k-mer lists and writes 121 gzip tables,
# the largest of them possibly over 100 million k-mers. Asked: 64 GB, 8 h.
#SBATCH --job-name=nb361-svw-score
#SBATCH --partition=normal,hns
#SBATCH --account=ayeletv
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=/scratch/users/olgabot/nb361-pec/logs/%x.%j.out
set -euo pipefail
S=/scratch/users/olgabot/nb361-pec
R=/scratch/users/olgabot/2024-kmerseek-analysis-nb361/analysis/pec-domen2026-splash
W=${W:-$S/sets_vs_weighted}
export PATH=$S/env/bin:$PATH
/usr/bin/time -v -o $W/score.time python $R/scripts/sets_vs_weighted_score.py $W $R $W/results
