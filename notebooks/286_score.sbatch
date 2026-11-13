#!/bin/bash
#SBATCH --job-name=286-score
#SBATCH --account=ayeletv
#SBATCH --partition=hns
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=286-score-%j.out
#
# Notebook 286: score every path Ced9 -> X -> Y (and P66, and the 40 copies), rank BCL2 and
# CD47, write <NB286_DIR>/results/. Submitted after the step 2 arrays by
# `286_intermediate_sequence.py plan --submit`; can also be run by hand at any time on the
# step 2 settings finished so far (results/286_settings_done.json lists them).
#
# Sizing: per query, about 2_000 intermediates x about 6_000 step 2 regions each at E < 100
# (Ced9 has 3_079 such regions over 152 settings at 280 aa; the human median is 427 aa),
# about 12 M rows, 1-2 GB in polars; 42 queries read one after another. No earlier run.
set -euo pipefail

: "${SCRATCH:?}" "${NB286_DIR:?}"
PY=$HOME/venvs/nb283/bin/python
module load python/3.12.1
cd "$SLURM_SUBMIT_DIR"
$PY 286_intermediate_sequence.py score
echo "286: score done. Peak memory and time: sacct -j $SLURM_JOB_ID -o JobID,MaxRSS,Elapsed"
