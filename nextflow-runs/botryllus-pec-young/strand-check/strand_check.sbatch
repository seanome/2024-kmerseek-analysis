#!/bin/bash
#SBATCH --job-name=pec-strand-check
#SBATCH --partition=hns,normal
#SBATCH --account=ayeletv
#SBATCH --cpus-per-task=6
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=%x.%j.out
# Which strand is read 1 on? Salmon library-type detection on the first 1 million read
# pairs of three deep young PEC libraries, against the FullNames transcripts index the
# germline test built (same genome, same GTF). Index: 3.2 GB peak in that test.
set -euo pipefail
T=/scratch/users/olgabot/botryllus-shazam-PEC-20261023/nfcore/results/trimgalore
TX=/scratch/users/olgabot/botryllus-rnaseq-preprocess-test/work/bf/2677bd3c2c26d846ff97711b8b959f/genome.transcripts.fa
IDX=$PWD/tx_index
IMG=$SCRATCH/apptainer-cache/quay.io-biocontainers-salmon@sha256-3938cc6dfaca6f7ee14eac0cdc0f305fdff8faa7f14541c72e684feb1b443a74.img
apptainer exec -B /scratch $IMG salmon --version
# Transcripts only, no genome decoys: enough to see which strand read 1 is on.
[ -s $IDX/versionInfo.json ] || apptainer exec -B /scratch $IMG salmon index -t $TX -i $IDX -p 6
for s in SRR32233964 SRR32233986 SRR32233978; do
  { zcat $T/${s}_trimmed_1_val_1.fq.gz || true; } | head -n 4000000 | gzip -1 > $s.R1.fq.gz
  { zcat $T/${s}_trimmed_2_val_2.fq.gz || true; } | head -n 4000000 | gzip -1 > $s.R2.fq.gz
  apptainer exec -B /scratch $IMG salmon quant --threads 6 --index $IDX -l A \
    -1 $s.R1.fq.gz -2 $s.R2.fq.gz --skipQuant -o $s
  cp $s/lib_format_counts.json $s.lib_format_counts.json
done
