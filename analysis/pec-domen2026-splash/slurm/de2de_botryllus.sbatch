#!/bin/bash
# Botryllus side of the DE-to-DE test (design/de2de/README.md). One task per contrast: the reads
# holding KaMRaT + DESeq2 p < 0.05 k-mers, split by direction, framed with Botryllus protein
# 10-mers, then the distinct protein k7, Dayhoff k13 and HP k25 k-mers of each direction.
# Writes $S/de2de/botryllus/<contrast>/{up,down}.codes.tsv.
# Resources: held reads are at most 1,015,040 (young 132 d); best-frame on 64 Botryllus runs of
# ~11 M read pairs took minutes, so 1 M reads is small. 16 GB, 2 h.
#SBATCH --job-name=nb361-de2de-bot
#SBATCH --partition=normal,hns
#SBATCH --account=ayeletv
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --exclude=sh03-13n20
#SBATCH --array=0-2
#SBATCH --output=/scratch/users/olgabot/nb361-pec/logs/%x.%A_%a.out
set -euo pipefail
S=/scratch/users/olgabot/nb361-pec
R=/scratch/users/olgabot/2024-kmerseek-analysis-nb361/analysis/pec-domen2026-splash
export PATH=$S/env/bin:$PATH
(cd $R/pec-tools && cargo build --release 2>&1 | tail -1)
PT=$R/pec-tools/target/release/pec-tools
contrasts=(young_13_49d young_132d old_41_93d)
c=${contrasts[$SLURM_ARRAY_TASK_ID]}
held=$S/contrasts/$c/search_kamrat_deseq2_p05_reads/held.reads.tsv
O=$S/de2de/botryllus/$c
mkdir -p $O
for d in up down; do
  # FASTQ with the read as its own quality line: best-frame reads only the sequence line
  awk -F'\t' -v d=$d 'NR>1 && $2==d {print "@r"NR"\n"$1"\n+\n"$1}' $held > $O/$d.fastq
  $PT best-frame $S/db/new_genome/protein.fasta protein 10 $O/$d 0 orpheum 6 < $O/$d.fastq 2> $O/$d.best_frame.log
  $PT kmer-codes protein:7,dayhoff:13,hp:25 < $O/$d.ge0.faa > $O/$d.codes.tsv
  command rm -f $O/$d.fastq
done
wc -l $O/*.codes.tsv
