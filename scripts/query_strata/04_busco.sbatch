#!/bin/bash
#SBATCH --job-name=qs-busco
#SBATCH --account=ayeletv
#SBATCH --partition=hns,normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/users/olgabot/leaderboard-query-strata/logs/%x-%j.log
#
# 6. Gene-model completeness: BUSCO in protein mode on the whole QfO 2020_04 human proteome
# (20,600 proteins, the proteome the 998 queries were drawn from), primates_odb12, the
# closest lineage to human. Each query's status is read from full_table.tsv afterwards.
# Ask: 8 CPUs, 16 GB, 4 h. No measured history; the peak is read with sacct afterwards.
set -euo pipefail
source /scratch/users/olgabot/leaderboard-query-strata/repo/scripts/query_strata/paths.sh
OUT=$WD/out/busco
mkdir -p "$OUT"
cd "$OUT"
apptainer exec --bind "$WD" --bind "$QFO_PIPE" "$BUSCO_IMG" busco --version | tee busco_version.txt
start=$(date +%s)
apptainer exec --bind "$WD" --bind "$QFO_PIPE" "$BUSCO_IMG" busco \
  -i "$HUMAN_PROTEOME" -m proteins -l "$BUSCO_LINEAGE" -c "$SLURM_CPUS_PER_TASK" \
  -o busco_human --out_path "$OUT" --offline --download_path "$WD/ref/busco_downloads" -f
echo "busco wall seconds: $(( $(date +%s) - start ))"
find "$OUT/busco_human" -maxdepth 3 -name 'full_table.tsv' -o -maxdepth 3 -name 'short_summary*.txt' | sort
