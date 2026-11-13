#!/bin/bash
# Shuffled list proteins searched against the same run sketches as job 50793573, at the same six
# settings, cutoff 0.1, then the reach table. Resources: job 50793573 took 67 min and 14 GB on 8 CPUs
# for 1,698 queries at two cutoffs; this is half as many queries at one cutoff.
#SBATCH --job-name=nb361-sra-alphabets
#SBATCH --partition=normal,hns
#SBATCH --account=ayeletv
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=/scratch/users/olgabot/nb361-pec/logs_kamrat_reads/%x.%j.out
set -euo pipefail
S=/scratch/users/olgabot/nb361-pec
R=/scratch/users/olgabot/2024-kmerseek-analysis-nb361/analysis/pec-domen2026-splash
W=$S/sra_search
export PATH=$S/env/bin:$PATH
mkdir -p $W/search_shuffled && cd $W/search_shuffled
python $R/scripts/sra_shuffled_queries.py $W/queries/queries.faa $W/queries/query_map.tsv shuffled.faa
printf 'name,genome_filename,protein_filename\nqueries,,%s\n' "$PWD/shuffled.faa" > q.csv
sourmash scripts manysketch q.csv --singleton -c $SLURM_CPUS_PER_TASK \
  -p hp,k=27,scaled=10 -p hp,k=30,scaled=10 -p dayhoff,k=12,scaled=10 -p dayhoff,k=13,scaled=10 \
  -p protein,k=6,scaled=10 -p protein,k=7,scaled=10 -o queries.sig.zip > queries.sketch.log 2>&1
sourmash sig describe --csv queries.describe.csv queries.sig.zip > /dev/null 2>&1
for spec in hp:27 hp:30 dayhoff:12 dayhoff:13 protein:6 protein:7; do
  m=${spec%:*}; k=${spec#*:}
  o=ms.ge0.1.$m.k$k.csv
  [ -s "$o" ] && continue
  sourmash scripts manysearch queries.sig.zip $W/search_all/against.ge0.1.txt -m $m -k $k -s 10 -t 0 -A --ignore-abundance -N \
    -c $SLURM_CPUS_PER_TASK -o $o.part > $o.log 2>&1
  mv $o.part $o
done
python $R/scripts/sra_alphabet_reach.py $W/search_all $W/search_shuffled $W/queries/query_map.tsv $W/runs.tsv $W/alphabet_reach
