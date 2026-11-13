#!/bin/bash
# Botryllus up and down k-mer sets at the 11 kmerseek settings of design/sets_vs_weighted/settings.tsv
# (rules: design/sets_vs_weighted/decision_rules.md). One task per row of botryllus_sets.tsv: the 3 real
# comparisons and the 7 swaps between pairs of colonies. Same steps as de2de_botryllus.sbatch: the
# reads holding KaMRaT + DESeq2 p < 0.05 k-mers, split by direction, framed with Botryllus protein
# 10-mers, then the distinct k-mers of each direction.
# Writes $W/botryllus/<name>/{up,down}.codes.tsv.
# Resources: de2de_botryllus.sbatch asked 16 GB and 2 h for 3 settings on the same reads (at most
# 1,015,040 held reads) and finished in minutes. 11 settings hold about 4 times as many k-mers.
#SBATCH --job-name=nb361-svw-bot
#SBATCH --partition=normal,hns
#SBATCH --account=ayeletv
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --exclude=sh03-13n20
#SBATCH --array=0-9
#SBATCH --output=/scratch/users/olgabot/nb361-pec/logs/%x.%A_%a.out
set -euo pipefail
S=/scratch/users/olgabot/nb361-pec
R=/scratch/users/olgabot/2024-kmerseek-analysis-nb361/analysis/pec-domen2026-splash
W=${W:-$S/sets_vs_weighted}
export PATH=$S/env/bin:$PATH
# several tasks start together; one builds, under a lock
mkdir -p $W
(flock 9; cd $R/pec-tools && cargo build --release 2>&1 | tail -1) 9> $W/build.lock
PT=$R/pec-tools/target/release/pec-tools
SET=$(awk -F'\t' 'NR > 1 { printf "%s%s:%s", (NR > 2 ? "," : ""), $1, $2 }' $R/design/sets_vs_weighted/settings.tsv)
row=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" $R/design/sets_vs_weighted/botryllus_sets.tsv)
IFS=$'\t' read -r name comparison labels dir <<< "$row"
held=$S/$dir/search_kamrat_deseq2_p05_reads/held.reads.tsv
O=$W/botryllus/$name
[ -e $O/codes.done ] && { echo "$name done"; exit 0; }
mkdir -p $O
echo "$name: settings $SET"
for d in up down; do
  # FASTQ with the read as its own quality line: best-frame reads only the sequence line
  awk -F'\t' -v d=$d 'NR>1 && $2==d {print "@r"NR"\n"$1"\n+\n"$1}' $held \
    | $PT best-frame $S/db/new_genome/protein.fasta protein 10 $O/$d 0 orpheum 6 2> $O/$d.best_frame.log
  $PT kmer-codes $SET < $O/$d.ge0.faa > $O/$d.codes.tsv
  command rm -f $O/$d.ge0.faa
done
wc -l $O/*.codes.tsv
touch $O/codes.done
