#!/bin/bash
#SBATCH -J expr-sketch
#SBATCH -p hns
#SBATCH -t 12:00:00
#SBATCH -c 8
#SBATCH --mem 64G
# One array task per row of runs_to_index.tsv (row 1 = smallest run).
#   sbatch --array=1-3,52 -o $WORK/logs/sketch.%A_%a.log sketch_runs.sbatch $WORK
# Memory and time asks come from the test rows' sacct (README, "Sizing").
set -uo pipefail
work=$1
here=/scratch/users/olgabot/2024-kmerseek-analysis-expression-index/analysis/expression-index
export RAYON_NUM_THREADS=$SLURM_CPUS_PER_TASK
"$here/sketch_run.sh" "$SLURM_ARRAY_TASK_ID" "$here/runs_to_index.tsv" "$work"
