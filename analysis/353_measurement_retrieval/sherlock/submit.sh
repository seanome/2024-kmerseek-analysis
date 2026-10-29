#!/bin/bash
# Submit notebook 353's kmerseek runs (workflow A) on Sherlock, after the compute budget is
# agreed and the inputs are copied (see README.md, "Workflow A on Sherlock").
# Run on a Sherlock login node, from inside the repository checkout:
#   bash analysis/353_measurement_retrieval/sherlock/submit.sh
set -euo pipefail
REPO=$(git rev-parse --show-toplevel)
S=$REPO/analysis/353_measurement_retrieval/sherlock
D=${NB353_DIR:-$SCRATCH/353_measurement_retrieval}
mkdir -p "$D/logs"
n=$(($(wc -l < "$D/in/kmerseek_settings.tsv") - 1))
echo "repo commit $(git rev-parse --short HEAD); $n kmerseek settings"
echo "asks: build 8 CPU 16G 1h; kmerseek x$n 4 CPU 16G 2h -> at most $((8 + n * 4 * 2)) CPU-hours"
EXP=ALL,NB353_REPO=$REPO,NB353_DIR=$D
b=$(sbatch --parsable --output="$D/logs/%x-%j.out" --export=$EXP "$S/01_build.sbatch")
k=$(sbatch --parsable --output="$D/logs/%x-%A_%a.out" --dependency=afterok:$b \
    --array=0-$((n - 1)) --export=$EXP "$S/02_index_search.sbatch")
echo "build $b, kmerseek array $k"
echo "Watch: squeue -u \$USER -o '%.12i %.24j %.8T %.10M %.6m %R' | grep 353"
