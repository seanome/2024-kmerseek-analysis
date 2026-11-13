#!/bin/bash
# Human side of the DE-to-DE test (design/de2de/README.md). One task per run of
# design/de2de/human_runs.tsv: frame the run's reads with GENCODE protein 10-mers, then count every
# Botryllus up or down k-mer in them. Writes $S/de2de/human/<run>.counts.tsv.gz.
# W=$S/de2de_swap (sbatch --export=ALL,W=...) counts the label-swap k-mers instead.
# First task merges the Botryllus k-mer lists into $S/de2de/query_codes.tsv; the others wait for it.
# Resources: framing a human run plus sketching took at most 1 h 28 min and 9.0 GB
# (sra_hp24_27_sketch.sbatch, GSE99963 and GSE144304). GSE120862 runs are about 2 GB gzipped,
# 2-3 times the others, so 8 h. The query tables add a few GB: 24 GB.
#SBATCH --job-name=nb361-de2de-hum
#SBATCH --partition=normal,hns
#SBATCH --account=ayeletv
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=8:00:00
#SBATCH --exclude=sh03-13n20
#SBATCH --array=0-167%40
#SBATCH --output=/scratch/users/olgabot/nb361-pec/logs/%x.%A_%a.out
set -euo pipefail
S=/scratch/users/olgabot/nb361-pec
R=/scratch/users/olgabot/2024-kmerseek-analysis-nb361/analysis/pec-domen2026-splash
W=${W:-$S/de2de}
PT=$R/pec-tools/target/release/pec-tools
Q=$W/query_codes.tsv
if [ ! -s $Q ]; then
  # several tasks may start together; one writes the file, under a lock
  (
    flock 9
    if [ ! -s $Q ]; then
      { echo -e "alphabet\tk\tcode"; tail -q -n +2 $W/botryllus/*/{up,down}.codes.tsv | sort -u; } > $Q.tmp
      mv $Q.tmp $Q
    fi
  ) 9> $W/query_codes.lock
fi
row=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" $R/design/de2de/human_runs.tsv)
series=${row%%$'\t'*}
run=${row##*$'\t'}
out=$W/human/$run.counts.tsv.gz
mkdir -p $W/human
[ -s $out ] && { echo "$run done"; exit 0; }
L=${L_SCRATCH:-/tmp}/nb361_de2de_$run
mkdir -p $L
trap 'command rm -rf "$L"' EXIT
/usr/bin/time -v -o $W/human/$run.time bash -c "set -o pipefail; zcat $S/human/trimmed/$series/$run.fastq.gz | $PT best-frame $S/db/gencode_v50/gencode.v50.pc_translations.fa protein 10 $L/$run 0 orpheum 6" 2> $W/human/$run.best_frame.log
$PT count-kmers $Q $L/$run.counts.tsv < $L/$run.ge0.faa
gzip -c $L/$run.counts.tsv > $out.tmp
mv $out.tmp $out
