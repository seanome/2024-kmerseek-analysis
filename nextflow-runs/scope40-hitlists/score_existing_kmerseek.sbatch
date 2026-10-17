#!/bin/bash
# Score the kmerseek SCOPe40 pair tables that already exist on Sherlock, per query, with
# bin/score_hitlist.py (Foldseek's benchmark rules). No search runs here.
#
# The tables are the one-row-per-pair outputs the scope40-alphabet-ksize-v040 combination
# runs kept in their work directories (PR 107): 8 kmerseek settings at the usual k range
# (runs/combine), 8 below it (runs/combine-lowk), and the same 16 on dipeptide-shuffled
# queries (runs/*-shuffled-k2-seed1). Ranked by region_evalue, smaller first.
#
# From a clone of this branch on Sherlock: first list the tables (a find, on the login node),
# then submit one array task per table:
#   bash nextflow-runs/scope40-hitlists/score_existing_kmerseek.sbatch
#   sbatch --array=0-31 nextflow-runs/scope40-hitlists/score_existing_kmerseek.sbatch
# Memory: the largest table (dayhoff6 k6, 8.6 GB of parquet) is the reason for 128 GB.
#SBATCH --job-name=s40-score-kmerseek
#SBATCH --partition=hns,normal
#SBATCH --account=ayeletv
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=logs/s40-score-kmerseek.%A_%a.log
set -euo pipefail

RUNS=${RUNS:-$SCRATCH/2024-kmerseek-scope40-v040/nextflow-runs/scope40-alphabet-ksize-v040/runs}
DOMAINS=${DOMAINS:-$SCRATCH/scope/results-scope40-alphabet-ksize-v040/scope_domains.tsv}
OUT=${OUT:-$SCRATCH/scope/results-scope40-hitlists/kmerseek}
PY=${PY:-$SCRATCH/venvs/nb275/bin/python}
HERE=$(cd "$(dirname "$0")" && pwd)
[ -f "$HERE/bin/score_hitlist.py" ] || HERE=$SLURM_SUBMIT_DIR/nextflow-runs/scope40-hitlists
mkdir -p "$OUT" logs

# One table per (run, file name): the first copy in a sorted listing of the work directory.
list=$OUT/tables.txt
if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
  for run in combine combine-lowk combine-shuffled-k2-seed1 combine-lowk-shuffled-k2-seed1; do
    # Only tables written by a task that exited 0 (a retried task leaves its failed copy).
    find "$RUNS/$run/work" -maxdepth 3 -name '*.pairs.parquet' -type f | sort \
      | while read -r f; do [ "$(cat "$(dirname "$f")/.exitcode" 2>/dev/null)" = 0 ] && echo "$f"; done \
      | awk -v r="$run" -F/ '{n=$NF; if (!(n in seen)) {seen[n]=1; print r "\t" $0}}'
  done > "$list.tmp"
  mv "$list.tmp" "$list"
  echo "$(wc -l < "$list") tables listed in $list; submit --array=0-$(( $(wc -l < "$list") - 1 ))"
  exit 0
fi
ml python/3.12.1
line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$list")
run=$(cut -f1 <<< "$line")
table=$(cut -f2 <<< "$line")
setting=$(basename "$table" .pairs.parquet)
case "$run" in *shuffled*) tool=kmerseek_shuffled ;; *) tool=kmerseek ;; esac
echo "$(date -Is) $run $setting $table"
echo "commit $(git -C "$HERE" rev-parse HEAD 2>/dev/null || echo unknown)"

/usr/bin/time -v "$PY" "$HERE/bin/score_hitlist.py" --domains "$DOMAINS" --hits "$table" \
  --score region_evalue --lower-first --tool "$tool" --setting "$setting" \
  --out-prefix "$OUT/$tool.$setting"
echo "$(date -Is) done"
