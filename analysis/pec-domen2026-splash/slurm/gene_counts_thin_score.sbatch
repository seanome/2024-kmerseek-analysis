#!/bin/bash
# Score each thinned gene-counts run, then the swapped-label designs.
#SBATCH --job-name=nb361-gc-tscore
#SBATCH --partition=normal,hns
#SBATCH --account=ayeletv
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=6:00:00
#SBATCH --output=/scratch/users/olgabot/nb361-pec/cheap_route/thin/score.%j.out
set -euo pipefail
S=/scratch/users/olgabot/nb361-pec
W=$S/cheap_route
R=/scratch/users/olgabot/2024-kmerseek-analysis-nb361/analysis/pec-domen2026-splash
export PATH=$S/env/bin:$PATH PYTHONPATH=$R/scripts
python $R/scripts/nfcore_list_sizes.py $S/nfcore/differentialabundance $S/paper/paper_deseq_all.parquet $S/nfcore/score_with_counts/list_sizes.tsv
for N in 10 100 1000; do
  python $R/scripts/cheap_route_score.py $S $W/thin/old.$N/all.counts.tsv $R/design $S/db/botznik.id_map.tsv $W/thin/old.$N/score sample_mean
  d=$W/thin/new.$N
  mkdir -p $d/directions
  for c in young_2h young_24h old_24h young_13_49d young_132d old_41_93d; do
    python $R/scripts/gene_directions.py $d/all.counts.tsv $R/design/contrast_$c.design.tsv $d/directions/$c.tsv sample_mean
  done
  python $R/scripts/nfcore_recovery.py $S/contrasts $S/nfcore/differentialabundance $S/db/proteome.id_map.tsv $d/score $d/directions
done
python $R/scripts/gene_counts_swaps.py $W/new_proteome/all.counts.tsv $W/all_reads/all.counts.tsv $R $S/nfcore/differentialabundance $S/nfcore/design $S/paper/paper_named_lists.tsv $S/db/botznik.id_map.tsv $W/swaps 16
