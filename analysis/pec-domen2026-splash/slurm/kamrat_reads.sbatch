#!/bin/bash
# KaMRaT contigs replaced with the reads that hold their k-mers, on the three small contrasts
# (notebook 361). Each task takes the kept contigs of one label set and route, finds their reads
# (pec-tools reads-holding), then translates and searches as 16_kamrat_nominal.sbatch does, against
# the old gene models only: the paper's lists are scored on those.
# Tasks: t-test and DESeq2 on real labels, and DESeq2 on every swap of these contrasts.
# Resources: young_24h's 3.19 M peptides took 128 GB and 58 min on 16 CPUs across all three
# protein sets (16_kamrat_nominal.sbatch). 8 h covers about 8 times as many peptides.
#SBATCH --job-name=nb361-kamrat-reads
#SBATCH --partition=normal,hns
#SBATCH --account=ayeletv
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=8:00:00
#SBATCH --array=0-12
#SBATCH --output=/scratch/users/olgabot/nb361-pec/logs_kamrat_reads/%x.%A_%a.out
set -euo pipefail
S=/scratch/users/olgabot/nb361-pec
R=/scratch/users/olgabot/2024-kmerseek-analysis-nb361/analysis/pec-domen2026-splash
export PATH=$S/env/bin:$PATH
PT=$S/pec-tools/target/release/pec-tools
N=$SLURM_CPUS_PER_TASK
TASKS=()
for c in young_13_49d young_132d old_41_93d; do
  TASKS+=("$c $S/contrasts/$c kamrat_p05" "$c $S/contrasts/$c kamrat_deseq2_p05")
done
for key in $(awk -F'\t' 'NR > 1 && $2 ~ /^(young_13_49d|young_132d|old_41_93d)$/ { print $1 }' $R/design/label_swap_kamrat/manifest.tsv); do
  TASKS+=("${key%/*} $S/label_swap/$key kamrat_deseq2_p05")
done
[ ${#TASKS[@]} -eq 13 ]
set -- ${TASKS[$SLURM_ARRAY_TASK_ID]}
C=$1; D=$2; ROUTE=$3
OUT=$D/search_${ROUTE}_reads
[ -e "$OUT/search.done" ] && { echo "$OUT already done"; exit 0; }
W=$(mktemp -d "$S/tmp_kamrat_reads_XXXX")
cd "$W"
/usr/bin/time -v -o reads.time "$PT" reads-holding "$D/$ROUTE/contigs.tsv" "$S/contrasts/$C/input.txt" held 31 100 2> reads.log
cat reads.log
ln -s held.reads.tsv seqs.tsv
"$PT" translate seqs.tsv pep 20 read 2> translate.log
cat translate.log
rm seqs.tsv
seqkit split2 -s 500000 -O chunks pep.peptides.faa 2> split.log
for fa in chunks/*.faa; do
  c=$(basename "$fa" .faa); c=${c##*.}
  sourmash scripts manysketch <(printf 'name,genome_filename,protein_filename\n%s,,%s\n' "$c" "$PWD/$fa") \
    --singleton -p protein,k=7,scaled=5 -p dayhoff,k=12,scaled=5 -p hp,k=30,scaled=5 -c "$N" -o "$c.scaled5.zip"
  for spec in protein:7 dayhoff:12 hp:30; do
    m=${spec%:*}; k=${spec#*:}
    for s in 10 100 5; do
      sourmash scripts manysearch "$c.scaled5.zip" "$S/db/botznik.$m.k$k.s$s.rocksdb" -m $m -k $k -s $s \
        -t 0.000001 -N -c "$N" -o "ms.$c.botznik.$m.k$k.s$s.csv" 2>> manysearch.log
    done
  done
  rm "$c.scaled5.zip"
done
python "$R/scripts/make_pairs.py" pairs.tsv hits.parquet ms.*.csv
rm ms.*.csv
python "$R/scripts/pair_align.py" pairs.botznik.tsv pep.peptides.faa "$S/db/botznik.faa" align.botznik.parquet "$N"
touch search.done
mv "$W" "$OUT"
