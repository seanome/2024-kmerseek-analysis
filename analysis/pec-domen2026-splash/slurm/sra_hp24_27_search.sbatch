#!/bin/bash
# List, random and shuffled proteins against every run at HP k = 24, 25, 26, 27, scaled 1, then the reach
# table. Resources: the HP search at scaled 10 took 6 min and 0.93 GB on 8 CPUs for 3 settings;
# scaled 1 holds about 10 times the hashes and there are half again as many queries and 4 settings.
#SBATCH --job-name=nb361-hp2427-search
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
python $R/scripts/sra_shuffled_queries.py $W/queries/queries.faa $W/queries/query_map.tsv $W/hp24_27/shuffled.faa
ls $W/hp24_27/*/*.sig.zip > $W/hp24_27/against.txt
[ "$(wc -l < $W/hp24_27/against.txt)" = 172 ]
for set in real shuffled; do
  mkdir -p $W/hp24_27_search/$set && cd $W/hp24_27_search/$set
  faa=$W/queries/queries.faa; [ $set = shuffled ] && faa=$W/hp24_27/shuffled.faa
  printf 'name,genome_filename,protein_filename\nqueries,,%s\n' "$faa" > q.csv
  sourmash scripts manysketch q.csv --singleton -c $SLURM_CPUS_PER_TASK \
    -p hp,k=24,scaled=1 -p hp,k=25,scaled=1 -p hp,k=26,scaled=1 -p hp,k=27,scaled=1 -o queries.sig.zip > queries.sketch.log 2>&1
  sourmash sig describe --csv queries.describe.csv queries.sig.zip > /dev/null 2>&1
  [ $set = real ] && sourmash sig describe --csv runs.ge0.describe.csv $(cat $W/hp24_27/against.txt) > /dev/null 2>&1
  for k in 24 25 26 27; do
    o=ms.ge0.hp.k$k.csv
    [ -s "$o" ] && continue
    /usr/bin/time -v -o $o.time sourmash scripts manysearch queries.sig.zip $W/hp24_27/against.txt -m hp -k $k -s 1 -t 0 -A --ignore-abundance -N \
      -c $SLURM_CPUS_PER_TASK -o $o.part > $o.log 2>&1
    mv $o.part $o
  done
done
python $R/scripts/sra_alphabet_reach.py $W/hp24_27_search/real $W/hp24_27_search/shuffled $W/queries/query_map.tsv $W/runs.tsv $W/hp24_27_reach hp:24,hp:25,hp:26,hp:27 ge0
