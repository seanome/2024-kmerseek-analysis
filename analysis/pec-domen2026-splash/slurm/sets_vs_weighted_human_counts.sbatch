#!/bin/bash
# Version A, human side (design/sets_vs_weighted/decision_rules.md). One task per run of
# design/sets_vs_weighted/human_runs.tsv (192 runs: every run of the 11 comparisons, and the 14 more
# that only PR 201's Fig. 4 groupings use, for the check against PR 201): frame the run's reads with
# GENCODE protein 10-mers, then count every Botryllus up or down k-mer (10 sets, 11 settings) in
# them. Same steps as de2de_human.sbatch.
# Writes $W/human/<run>.counts.tsv.gz, and keeps the framed peptides in $W/human_framed/<run>.faa.gz
# so the later count of every human k-mer (table sizes) does not frame the reads again.
# First task merges the Botryllus k-mer lists into $W/query_codes.tsv; the others wait for it.
# Run after sets_vs_weighted_botryllus.sbatch (afterok).
# Resources, measured on de2de_human.sbatch (jobs 51590632 and 51658368, 168 runs each): at most
# 19 min 16 s and 1.8 GB with 4.8 to 9.8 million query k-mers at 3 settings. Here about 4 times the
# settings and all 10 sets in one list, so about 40 million query k-mers: 16 GB, 3 h.
#SBATCH --job-name=nb361-svw-hum
#SBATCH --partition=normal,hns
#SBATCH --account=ayeletv
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=3:00:00
#SBATCH --exclude=sh03-13n20
#SBATCH --array=0-191%40
#SBATCH --output=/scratch/users/olgabot/nb361-pec/logs/%x.%A_%a.out
set -euo pipefail
S=/scratch/users/olgabot/nb361-pec
R=/scratch/users/olgabot/2024-kmerseek-analysis-nb361/analysis/pec-domen2026-splash
W=${W:-$S/sets_vs_weighted}
PT=$R/pec-tools/target/release/pec-tools
Q=$W/query_codes.tsv
n_sets=$(($(wc -l < $R/design/sets_vs_weighted/botryllus_sets.tsv) - 1))
[ $(ls $W/botryllus/*/codes.done | wc -l) -eq $n_sets ] || { echo "not all $n_sets Botryllus sets are done"; exit 1; }
if [ ! -s $Q ]; then
  # several tasks may start together; one writes the file, under a lock
  (
    flock 9
    if [ ! -s $Q ]; then
      { echo -e "alphabet\tk\tcode"; tail -q -n +2 $W/botryllus/*/{up,down}.codes.tsv | LC_ALL=C sort -u -S 4G -T $W; } > $Q.tmp
      mv $Q.tmp $Q
    fi
  ) 9> $W/query_codes.lock
fi
row=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" $R/design/sets_vs_weighted/human_runs.tsv)
series=${row%%$'\t'*}
run=${row##*$'\t'}
out=$W/human/$run.counts.tsv.gz
mkdir -p $W/human $W/human_framed
[ -s $out ] && { echo "$run done"; exit 0; }
L=${L_SCRATCH:-/tmp}/nb361_svw_$run
mkdir -p $L
trap 'command rm -rf "$L"' EXIT
/usr/bin/time -v -o $W/human/$run.time bash -c "set -o pipefail; zcat $S/human/trimmed/$series/$run.fastq.gz | $PT best-frame $S/db/gencode_v50/gencode.v50.pc_translations.fa protein 10 $L/$run 0 orpheum 6" 2> $W/human/$run.best_frame.log
/usr/bin/time -v -o $W/human/$run.count.time $PT count-kmers $Q $L/$run.counts.tsv < $L/$run.ge0.faa
gzip -1 -c $L/$run.ge0.faa > $W/human_framed/$run.faa.gz.tmp
mv $W/human_framed/$run.faa.gz.tmp $W/human_framed/$run.faa.gz
gzip -c $L/$run.counts.tsv > $out.tmp
mv $out.tmp $out
