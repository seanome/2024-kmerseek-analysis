#!/bin/bash
# One run of runs.tsv (array index = row): protein k7, Dayhoff k12 and k13 at scaled 1, on all reads with a frame.
# Frames as in 26_sra_k10_frame_sketch.sbatch: pec-tools best-frame, protein 10-mers of the species'
# own proteins. Its frame files were not kept, so frames are chosen again. Writes
# $S/sra_search/pd_k10/<run>/. Same frames as sra_hp24_27_sketch.sbatch, so
# every setting is sketched the same way.
# Resources: sra_hp24_27_sketch.sbatch took at most 1 h 28 min and 9.0 GB (jobs 51358172, 51553711);
# protein k-mers collide less than HP ones, so up to 1.5 times the hashes: 24 GB, 6 h. Node sh03-13n20
# ran frames 60 times slower there, so it is left out.
#SBATCH --job-name=nb361-pd-sketch
#SBATCH --partition=normal,hns
#SBATCH --account=ayeletv
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=6:00:00
#SBATCH --exclude=sh03-13n20
#SBATCH --array=0-171%32
#SBATCH --output=/scratch/users/olgabot/nb361-pec/logs_kamrat_reads/%x.%A_%a.out
set -euo pipefail
S=/scratch/users/olgabot/nb361-pec
W=$S/sra_search
export PATH=$S/env/bin:$PATH
cd "$W"
run=$(awk -F'\t' 'NR>1{print $1}' runs.tsv | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")
dataset=$(awk -F'\t' -v r="$run" '$1==r{print $3}' runs.tsv)
files=$(awk -F'\t' -v r="$run" '$1==r{print $9}' runs.tsv | tr ';' ' ')
if [ "$dataset" = PRJNA1219309 ]; then ref=$S/db/new_genome/protein.fasta; else ref=$S/db/gencode_v50/gencode.v50.pc_translations.fa; fi
O=$W/pd_k10/$run
mkdir -p "$O"
[ -e "$O/sketch.done" ] && { echo "$run done"; exit 0; }
L=${L_SCRATCH:-/tmp}/nb361_pd_$run
mkdir -p "$L"
trap 'command rm -rf "$L"' EXIT
/usr/bin/time -v -o "$O/best_frame.time" bash -c "set -o pipefail; zcat $files | $S/pec-tools/target/release/pec-tools best-frame $ref protein 10 $L/$run 0 orpheum 6" 2> "$O/best_frame.log"
printf 'name,genome_filename,protein_filename\n%s,,%s\n' "$run" "$L/$run.ge0.faa" > "$L/ms.csv"
/usr/bin/time -v -o "$O/manysketch.time" sourmash scripts manysketch "$L/ms.csv" -c 1 \
  -p protein,k=7,scaled=1 -p dayhoff,k=12,scaled=1 -p dayhoff,k=13,scaled=1 -o "$O/$run.sig.zip" > "$O/manysketch.log" 2>&1
touch "$O/sketch.done"
