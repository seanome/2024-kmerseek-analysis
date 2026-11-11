#!/bin/bash
# Botryllus up/down rule with poscounts size factors in place of CPM (design/sample_scaling/README.md).
# Sized from jobs 51207129 and 51216691, which did the same scoring: under 30 s, 0.6 GB.
#SBATCH --job-name=nb361-scaling
#SBATCH --partition=normal,hns
#SBATCH --account=ayeletv
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=30:00
#SBATCH --output=/scratch/users/olgabot/nb361-pec/sample_scaling/score.%j.out
set -euo pipefail
S=/scratch/users/olgabot/nb361-pec
W=$S/cheap_route
O=$S/sample_scaling
R=/scratch/users/olgabot/2024-kmerseek-analysis-nb361/analysis/pec-domen2026-splash
export PATH=$S/env/bin:$PATH
mkdir -p $O/new_proteome/sample_mean $O/new_proteome/size_factor_mean
# the factors, checked against PyDESeq2, on both sets of counts
python $R/scripts/sample_scaling_factors.py $W/new_proteome/all.counts.tsv $R/design $O/factors_new_proteome.tsv
python $R/scripts/sample_scaling_factors.py $W/all_reads/all.counts.tsv $R/design $O/factors_old_proteome.tsv
# against nf-core's lists (new proteome); sample_mean again, as a check that 609 of 1,135 comes back
for rule in sample_mean size_factor_mean; do
  for c in young_2h young_24h old_24h young_13_49d young_132d old_41_93d; do
    python $R/scripts/gene_directions.py $W/new_proteome/all.counts.tsv $R/design/contrast_$c.design.tsv $O/new_proteome/$rule/$c.tsv $rule
  done
done
# nfcore_recovery.py names its two count routes gene_counts and gene_counts_mean: here they are
# sample_mean and size_factor_mean
python $R/scripts/nfcore_recovery.py $S/contrasts $S/nfcore/differentialabundance $S/db/proteome.id_map.tsv $O/nfcore $O/new_proteome/sample_mean $O/new_proteome/size_factor_mean
# against the paper's lists (old proteome)
python $R/scripts/cheap_route_score.py $S $W/all_reads/all.counts.tsv $R/design $S/db/botznik.id_map.tsv $O/paper_sample_mean sample_mean
python $R/scripts/cheap_route_score.py $S $W/all_reads/all.counts.tsv $R/design $S/db/botznik.id_map.tsv $O/paper_size_factor_mean size_factor_mean
