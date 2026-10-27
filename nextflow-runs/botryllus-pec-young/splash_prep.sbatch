#!/bin/bash
#SBATCH --job-name=pec-young-splash-prep
#SBATCH --partition=hns,normal
#SBATCH --account=ayeletv
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --array=1-52
#SBATCH --output=logs/%x.%A_%a.out
# One SPLASH read file per library: read 1 then read 2 of the Trim Galore pair, recompressed.
# Array task N takes row N of samplesheet.csv. The largest pair is 13 GB gzipped
# (SRR32233964, 69.9 million pairs); zcat plus 4-thread pigz holds a few MB in memory.
# Usage (from submit_all.sh): sbatch --chdir <run dir> splash_prep.sbatch <repo dir>
set -euo pipefail
REPO=$1
OUT=$SCRATCH/botryllus-pec-young/splash-reads
mkdir -p "$OUT"
IFS=, read -r sample r1 r2 _ < <(sed -n "$(( SLURM_ARRAY_TASK_ID + 1 ))p" "$REPO/samplesheet.csv")
f=$OUT/$sample.fq.gz
[ -s "$f.done" ] && { echo "$sample already done"; exit 0; }
zcat "$r1" "$r2" | pigz -p "$SLURM_CPUS_PER_TASK" -1 > "$f.part"
mv "$f.part" "$f"
echo "$(zcat "$f" | awk 'END{print NR/4}') reads" > "$f.done"
cat "$f.done"
