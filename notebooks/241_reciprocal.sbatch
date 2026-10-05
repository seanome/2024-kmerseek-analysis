#!/bin/bash
#SBATCH --job-name=241-reciprocal
#SBATCH --account=ayeletv
#SBATCH --partition=hns
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=241-reciprocal-%j.out
#
# Notebook 241, the reverse direction: BCL2 against the C. elegans proteome (where does
# CED-9 rank?) and CD47 against the B. burgdorferi proteome (where does P66 rank?), at
# every arm of the forward run. Then the forward collector again, which now also writes
# top_hits.csv (the 20 best human proteins for every arm, query and metric) from the
# search CSVs the forward run left in $NB241_DIR/search.
#
# Needs the forward run's outputs (plan.json, search/) and its kmerseek build. Submit
# from the same clone as 241_alphabet_ranking.sbatch:
#
#   cd $SCRATCH/241-alphabet-ranking-repo && git pull && sbatch notebooks/241_reciprocal.sbatch
#
# Sizing: same as the forward job. The C. elegans proteome (~26_000 proteins) is about
# the size of the human one and B. burgdorferi (1_291) is small, so this is about one
# forward run of work. The forward run took 40 minutes on a 16-core, 128 GB laptop with
# --workers 2. Its Sherlock peak memory was not read back: after this run, read both with
# `sacct -j <jobid> -o JobID,MaxRSS,Elapsed`.
#
# Copy the tables back with:
#
#   rsync -av sherlock-dtn:/scratch/users/olgabot/241-alphabet-ranking/top_hits.csv \
#       /Users/olga/data/botryllus/alphabet-ranking-three-cases/
#   rsync -av --include='*.csv' --include='*.json' --exclude='*' \
#       sherlock-dtn:/scratch/users/olgabot/241-alphabet-ranking/reciprocal/ \
#       /Users/olga/data/botryllus/alphabet-ranking-three-cases/reciprocal/
set -euo pipefail

: "${SCRATCH:?}" "${SLURM_SUBMIT_DIR:?}"
REPO=$SLURM_SUBMIT_DIR
export NB241_DIR=${NB241_DIR:-$SCRATCH/241-alphabet-ranking}
export NB241_HUMAN=${NB241_HUMAN:-$SCRATCH/data/gencode/human/v49/gencode.v49.pc_translations.canonical.fa}
export NB241_KMERSEEK=$SCRATCH/241-alphabet-ranking-kmerseek/target/release/kmerseek
IMG=$SCRATCH/apptainer-cache/docker.io-olgabot-kmerseek-2026-09-13-extend-chain.img

for f in "$NB241_KMERSEEK" "$NB241_HUMAN" "$NB241_DIR/plan.json" "$IMG"; do
    [ -s "$f" ] || { echo "241 reciprocal: missing input $f (run 241_alphabet_ranking.sbatch first)" >&2; exit 2; }
done
[ -n "$(ls "$NB241_DIR"/search/*.csv 2>/dev/null | head -1)" ] || \
    { echo "241 reciprocal: no forward search CSVs in $NB241_DIR/search" >&2; exit 2; }

module load gcc/14.2.0 python/3.12.1
"$NB241_KMERSEEK" --version

cd "$REPO/notebooks"
python3 241_reciprocal_driver.py --workers 2
apptainer exec --bind /scratch "$IMG" python3 241_reciprocal_driver.py --collect
apptainer exec --bind /scratch "$IMG" python3 241_alphabet_ranking_collect.py

echo "241 reciprocal: done. Peak memory and time: sacct -j $SLURM_JOB_ID -o JobID,MaxRSS,Elapsed"
