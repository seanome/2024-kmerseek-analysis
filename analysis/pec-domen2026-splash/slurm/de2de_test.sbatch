#!/bin/bash
# DE-to-DE test statistics (design/de2de/README.md, scripts/de2de_test.py), after de2de_human.sbatch.
# Resources: no history. Memory grows with the number of distinct Botryllus k-mers (a few arrays of
# 8 bytes per k-mer per setting); with up to 100 M per setting that is about 20 GB, so 64 GB.
# Each of 168 count files is read twice: 6 h.
#SBATCH --job-name=nb361-de2de-test
#SBATCH --partition=normal,hns
#SBATCH --account=ayeletv
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=/scratch/users/olgabot/nb361-pec/logs/%x.%j.out
set -euo pipefail
S=/scratch/users/olgabot/nb361-pec
R=/scratch/users/olgabot/2024-kmerseek-analysis-nb361/analysis/pec-domen2026-splash
export PATH=$S/env/bin:$PATH
/usr/bin/time -v -o $S/de2de/test.time python $R/scripts/de2de_test.py $S/de2de $R/design/human $R/design/de2de/human_runs.tsv $S/de2de/results
