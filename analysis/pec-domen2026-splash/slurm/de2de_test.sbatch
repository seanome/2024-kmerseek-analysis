#!/bin/bash
# DE-to-DE test statistics (design/de2de/README.md, scripts/de2de_test.py), after de2de_human.sbatch.
# W=$S/de2de_swap SETS=<set,...> (sbatch --export=ALL,...) runs the label-swap control.
# Resources: the 3 real contrasts took 39 s and 0.8 GB (job 51590633, 2026-11-08). Memory grows with
# the number of distinct Botryllus k-mers; the 7 swap sets hold at most about 3 times as many, so
# 16 GB and 1 h leave a wide margin.
#SBATCH --job-name=nb361-de2de-test
#SBATCH --partition=normal,hns
#SBATCH --account=ayeletv
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=/scratch/users/olgabot/nb361-pec/logs/%x.%j.out
set -euo pipefail
S=/scratch/users/olgabot/nb361-pec
R=/scratch/users/olgabot/2024-kmerseek-analysis-nb361/analysis/pec-domen2026-splash
W=${W:-$S/de2de}
export PATH=$S/env/bin:$PATH
/usr/bin/time -v -o $W/test.time python $R/scripts/de2de_test.py $W $R/design/human $R/design/de2de/human_runs.tsv $W/results ${SETS:-}
