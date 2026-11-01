#!/bin/bash
#SBATCH --job-name=286-step2
#SBATCH --account=ayeletv
#SBATCH --partition=hns
#SBATCH --output=286-step2-%A_%a.out
#
# Notebook 286, step 2: the whole human proteome against itself at one setting (array task
# i = row i of notebook 256's arms.json), keeping regions with E < 100. Submitted by
# `286_intermediate_sequence.py plan --submit` at the end of 286_intermediate_sequence.sbatch,
# with --array, --time, --mem and --cpus-per-task taken from the pilot (step2_plan.csv).
# A rerun resumes: finished chunks are kept until the setting's table is written.
set -euo pipefail

: "${SCRATCH:?}" "${SLURM_ARRAY_TASK_ID:?}" "${NB286_DIR:?}"
WORKERS=${NB286_WORKERS:-4}
export RAYON_NUM_THREADS=$((SLURM_CPUS_PER_TASK / WORKERS))
PY=$HOME/venvs/nb283/bin/python
module load python/3.12.1 gcc/14.2.0
cd "$SLURM_SUBMIT_DIR"
$PY 286_intermediate_sequence.py step2 --task "$SLURM_ARRAY_TASK_ID" --workers "$WORKERS"
