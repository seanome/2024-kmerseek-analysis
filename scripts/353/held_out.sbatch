#!/bin/bash
# Notebook 353, held-out half: the frozen setting (data/kill_tests/short_stretch/
# frozen_setting.json, committed before this is submitted) on every held-out record:
# about 204,000 self-retrieval queries and 657,000 shuffles.
#
# Submit the three chained jobs with one command, from the clone on Sherlock:
#
#   cd /scratch/users/olgabot/2024-kmerseek-analysis-nb353 && bash scripts/353/held_out.sbatch submit
#
# prep (index, queries, SEG, queries split in chunks) -> search array (one chunk per task) -> score,
# each started only when the one before it succeeded (afterok).
#SBATCH --account=ayeletv
#SBATCH --partition=hns,normal
#SBATCH --output=/scratch/users/olgabot/nb353/logs/%x.%A_%a.out
set -euo pipefail
repo=/scratch/users/olgabot/2024-kmerseek-analysis-nb353
root=/scratch/users/olgabot/nb353
frozen=$repo/data/kill_tests/short_stretch/frozen_setting.json
alphabet=$(python3 -c "import json;print(json.load(open('$frozen'))['alphabet'])")
k=$(python3 -c "import json;print(json.load(open('$frozen'))['k'])")

if [ "${1:-}" = submit ]; then
  cd "$repo"
  git log -1 --format='frozen setting from commit %h: %s' -- "$frozen"
  echo "setting: $alphabet k=$k"
  # Sizes from the development array (job 48681197): peak search memory 2.3 GB at most
  # (8 GB asked). The search runs on about one core, so time scales with queries: the
  # held-out half has 20.7 times the development queries. Chunks are sized from this
  # setting's own development search time so each takes about 30 min (1 h 30 asked).
  dev_s=$(python3 -c "
import re
t = open('$root/dev/$alphabet.k$k/search.time').read()
h = re.search(r'Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): (\S+)', t).group(1).split(':')
print(sum(float(x) * 60 ** i for i, x in enumerate(reversed(h))))")
  n_chunks=$(python3 -c "import math; print(max(4, math.ceil($dev_s * 20.7 / 1800)))")
  echo "development search took $dev_s s; held-out split into $n_chunks chunks"
  prep=$(sbatch --parsable --job-name=nb353-heldout-prep --cpus-per-task=2 --mem=8G --time=01:00:00 \
           --export=ALL,N_CHUNKS=$n_chunks "$0" step prep)
  srch=$(sbatch --parsable --job-name=nb353-heldout-search --array=0-$((n_chunks - 1)) --cpus-per-task=2 --mem=8G --time=01:30:00 \
           --dependency=afterok:$prep "$0" step search)
  scor=$(sbatch --parsable --job-name=nb353-heldout-score --cpus-per-task=4 --mem=48G --time=01:00:00 \
           --dependency=afterok:$srch "$0" step score)
  echo "prep $prep -> search $srch -> score $scor"
  exit 0
fi

[ "${1:-}" = step ] || { echo "usage: bash $0 submit" >&2; exit 2; }
cd "$repo"
case $2 in
  prep)   bash scripts/353/run_setting.sh held_out "$alphabet" "$k" "$root" prep ;;
  search) bash scripts/353/run_setting.sh held_out "$alphabet" "$k" "$root" search "$SLURM_ARRAY_TASK_ID" ;;
  score)  bash scripts/353/run_setting.sh held_out "$alphabet" "$k" "$root" score ;;
esac
