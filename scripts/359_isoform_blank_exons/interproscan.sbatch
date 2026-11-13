#!/bin/bash
# Notebook 359, step 3: InterProScan 5.78-109.0, Pfam only, CPU, on the ORFs of the
# coding isoforms that passed the SQANTI3 filter. For the sanity gate and the blank set.
#
# No measured InterProScan run on this cluster. proteome-interproscan/Makefile's estimate
# for all member databases is 1-2 s per protein per core; Pfam alone is a fraction of
# that. 32 CPUs and 24 h cover 1-2 x 10^5 ORFs at the all-database rate.
#SBATCH --job-name=359-interproscan
#SBATCH --account=ayeletv
#SBATCH --partition=hns
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x.%j.log

set -euo pipefail
WORK=$(pwd)
IPS=$WORK/data/interproscan-5.78-109.0
mkdir -p "$WORK/ips"
ml java/17.0.4
"$IPS/interproscan.sh" -i "$WORK/sqanti/orfs_pass_coding.fasta" -appl Pfam -f TSV -dp \
    -cpu 32 -T "${L_SCRATCH:-$WORK/ips}/ips_tmp" -o "$WORK/ips/pfam.tsv"
"$IPS/interproscan.sh" --version | head -3
wc -l "$WORK/ips/pfam.tsv"
