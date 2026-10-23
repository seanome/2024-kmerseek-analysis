#!/bin/bash
# Notebook 359, steps 7, 8 and 10: thresholds, the scan control, the three rules, the
# surviving landings. Reads only small tables (one best call per exon per setting), so a
# small ask: the scan bisection is the slow part, about 1 ms per exon side.
#SBATCH --job-name=359-score
#SBATCH --account=ayeletv
#SBATCH --partition=hns
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x.%j.log

set -euo pipefail
WORK=$(pwd)
IMG=/scratch/users/olgabot/apptainer-cache/docker.io-olgabot-kmerseek-0.4.1-rc1.img
apptainer exec --bind /scratch "$IMG" python3 code/scripts/359_isoform_blank_exons/score_landings.py \
    --build-dir "$WORK/build" --best-dir "$WORK/kmerseek/best" \
    --regions /scratch/users/olgabot/regions-index-pilot/results/regions_index/regions.parquet \
    --settings code/scripts/359_isoform_blank_exons/settings.tsv \
    --example-seed 359 --out-dir "$WORK/score"

# The tables the notebook reads, gathered for the copy back to the repo.
mkdir -p "$WORK/for_repo"
cp "$WORK"/score/{summary.csv,verdict.json,examples.json,survivors.parquet,landings.parquet,scan.parquet} "$WORK/for_repo/"
cp "$WORK"/build/{gate.json,build_summary.json,blank_exons.parquet} "$WORK/for_repo/"
cp "$WORK/data/manifest.tsv" "$WORK/data/tappas/header.txt" "$WORK/kmerseek/plan.tsv" "$WORK/for_repo/"
du -sh "$WORK/for_repo"
