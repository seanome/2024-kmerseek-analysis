#!/bin/bash
# List, random and shuffled proteins against every run at protein k7, Dayhoff k12 and k13, scaled 1,
# then the reach table. Resources: the HP k24-27 search at scaled 1 took 1 h 43 min and 15.6 GB on 8 CPUs
# (job 51553712); protein may hold up to 1.5 times the hashes.
#SBATCH --job-name=nb361-pd-search
#SBATCH --partition=normal,hns
#SBATCH --account=ayeletv
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=/scratch/users/olgabot/nb361-pec/logs_kamrat_reads/%x.%j.out
set -euo pipefail
S=/scratch/users/olgabot/nb361-pec
R=/scratch/users/olgabot/2024-kmerseek-analysis-nb361/analysis/pec-domen2026-splash
W=$S/sra_search
export PATH=$S/env/bin:$PATH
python $R/scripts/sra_shuffled_queries.py $W/queries/queries.faa $W/queries/query_map.tsv $W/pd_k10/shuffled.faa
ls $W/pd_k10/*/*.sig.zip > $W/pd_k10/against.txt
[ "$(wc -l < $W/pd_k10/against.txt)" = 172 ]
for set in real shuffled; do
  mkdir -p $W/pd_k10_search/$set && cd $W/pd_k10_search/$set
  faa=$W/queries/queries.faa; [ $set = shuffled ] && faa=$W/pd_k10/shuffled.faa
  printf 'name,genome_filename,protein_filename\nqueries,,%s\n' "$faa" > q.csv
  sourmash scripts manysketch q.csv --singleton -c $SLURM_CPUS_PER_TASK \
    -p protein,k=7,scaled=1 -p dayhoff,k=12,scaled=1 -p dayhoff,k=13,scaled=1 -o queries.sig.zip > queries.sketch.log 2>&1
  sourmash sig describe --csv queries.describe.csv queries.sig.zip > /dev/null 2>&1
  [ $set = real ] && sourmash sig describe --csv runs.ge0.describe.csv $(cat $W/pd_k10/against.txt) > /dev/null 2>&1
  for mk in protein:7 dayhoff:12 dayhoff:13; do
    m=${mk%:*}; k=${mk#*:}
    o=ms.ge0.$m.k$k.csv
    [ -s "$o" ] && continue
    /usr/bin/time -v -o $o.time sourmash scripts manysearch queries.sig.zip $W/pd_k10/against.txt -m $m -k $k -s 1 -t 0 -A --ignore-abundance -N \
      -c $SLURM_CPUS_PER_TASK -o $o.part > $o.log 2>&1
    mv $o.part $o
  done
done
python $R/scripts/sra_alphabet_reach.py $W/pd_k10_search/real $W/pd_k10_search/shuffled $W/queries/query_map.tsv $W/runs.tsv $W/pd_k10_reach protein:7,dayhoff:12,dayhoff:13 ge0
