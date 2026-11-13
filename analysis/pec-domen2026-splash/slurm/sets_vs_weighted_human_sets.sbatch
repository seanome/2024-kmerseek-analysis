#!/bin/bash
# Version B, human side (design/sets_vs_weighted/decision_rules.md). One task per row of
# design/sets_vs_weighted/human_sets.tsv: the 11 human comparisons and their 18 label swaps. The
# same steps as the Botryllus side: the reads holding the comparison's KaMRaT + DESeq2 p < 0.05
# k-mers (pec-tools reads-holding, 31-mers, at most 100 reads per contig and run), split by
# direction, framed with GENCODE protein 10-mers, then the distinct k-mers of each direction at the
# 11 settings.
# Writes $W/human_sets/<name>/{up,down}.<alphabet>.<k>.codes.gz (one code per line, no header; a
# setting too long for the reads has no file), reads.log, held.contigs.tsv and codes.done.
# Resources are scaled, not measured: reads-holding has not been run on human reads. The Botryllus
# runs kept 12 to 13 % of contigs x runs x 100 reads. The largest human comparison has 205,857
# contigs and 39 runs (803 million at 100 %, 700 million reads in all). At 35 % that is 280 million
# distinct reads, about 200 bytes each in memory: 56 GB. Framing ran at 148,000 reads per second
# (SRR7967699), so 280 million reads take about 32 min. Asked: 120 GB, 8 h.
# Measured since (job 51959235, 2026-11-11, all 29 tasks): at most 57 min 58 s and 20.4 GB.
#SBATCH --job-name=nb361-svw-hsets
#SBATCH --partition=normal,hns
#SBATCH --account=ayeletv
#SBATCH --cpus-per-task=2
#SBATCH --mem=120G
#SBATCH --time=8:00:00
#SBATCH --exclude=sh03-13n20
#SBATCH --array=0-28
#SBATCH --output=/scratch/users/olgabot/nb361-pec/logs/%x.%A_%a.out
set -euo pipefail
S=/scratch/users/olgabot/nb361-pec
R=/scratch/users/olgabot/2024-kmerseek-analysis-nb361/analysis/pec-domen2026-splash
W=${W:-$S/sets_vs_weighted}
export PATH=$S/env/bin:$PATH
mkdir -p $W
(flock 9; cd $R/pec-tools && cargo build --release 2>&1 | tail -1) 9> $W/build.lock
PT=$R/pec-tools/target/release/pec-tools
SET=$(awk -F'\t' 'NR > 1 { printf "%s%s:%s", (NR > 2 ? "," : ""), $1, $2 }' $R/design/sets_vs_weighted/settings.tsv)
row=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" $R/design/sets_vs_weighted/human_sets.tsv)
IFS=$'\t' read -r name comparison labels kind series paired dir design <<< "$row"
O=$W/human_sets/$name
[ -e $O/codes.done ] && { echo "$name done"; exit 0; }
mkdir -p $O
command rm -f $O/*.codes.gz
T=$(mktemp -d $W/tmp_human_sets_XXXX)
trap 'command rm -rf "$T"' EXIT
# sample and FASTQ of every run in the comparison's design
awk -F'\t' -v s=$S 'NR == FNR { series[$2] = $1; next } FNR > 1 { print $1 "\t" s "/human/trimmed/" series[$1] "/" $1 ".fastq.gz" }' \
  $R/design/sets_vs_weighted/human_runs.tsv $R/$design > $T/input.txt
[ $(wc -l < $T/input.txt) -eq $(($(wc -l < $R/$design) - 1)) ]
while IFS=$'\t' read -r sample fq; do [ -s "$fq" ] || { echo "missing $fq"; exit 1; }; done < $T/input.txt
echo "$name: $(wc -l < $T/input.txt) runs, settings $SET"
/usr/bin/time -v -o $O/reads.time $PT reads-holding $S/human/$dir/kamrat_deseq2_p05/contigs.tsv $T/input.txt $T/held 31 100 2> $O/reads.log
tail -1 $O/reads.log
for d in up down; do
  awk -F'\t' -v d=$d 'NR>1 && $2==d {print "@r"NR"\n"$1"\n+\n"$1}' $T/held.reads.tsv \
    | $PT best-frame $S/db/gencode_v50/gencode.v50.pc_translations.fa protein 10 $T/$d 0 orpheum 6 2> $O/$d.best_frame.log
  # one gzip file per setting, so the scoring step reads one setting at a time
  $PT kmer-codes $SET < $T/$d.ge0.faa \
    | awk -F'\t' -v o=$O/$d 'NR > 1 { f = "gzip > " o "." $1 "." $2 ".codes.gz"; print $3 | f; cmds[f] = 1 }
                             END { for (f in cmds) if (close(f) != 0) exit 1 }'
  command rm -f $T/$d.ge0.faa
done
command cp $T/held.contigs.tsv $O/held.contigs.tsv
ls -la $O
touch $O/codes.done
