#!/bin/bash
#SBATCH --job-name=275-seed-frontier
#SBATCH --account=ayeletv
#SBATCH --partition=hns
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=275-seed-frontier-%j.out
#
# Notebook 275, Figure 1 option 1: human proteins sharing an exact seed with 300 random
# human queries, real and shuffled, for hp_lehninger2, hp_thomas_dill2, dayhoff6 and
# protein20 at every k in scripts/run_275_seed_frontier.py.
#
# Submit from a checkout of olgabot/fig1-hp-runs on Sherlock. First time only:
#
#   git clone --branch olgabot/fig1-hp-runs https://github.com/seanome/2024-kmerseek-analysis $SCRATCH/2024-kmerseek-analysis-fig1
#
# Then, each run:
#
#   cd $SCRATCH/2024-kmerseek-analysis-fig1 && git pull && sbatch notebooks/275_seed_frontier.sbatch
#
# Python: Sherlock's python/3.12.1 module with numpy and polars in a venv under
# $SCRATCH/venvs/nb275, made by the job the first time (pip needs the compute node's internet).
# Wheels only, at notebook 283's versions: the newest numpy has no wheel for Sherlock and
# fails to build from source. That left an empty venv on 2026-10-13 (job 47673647), and the
# next job found its python and ran without numpy (47686608). So the venv is checked by
# importing both packages, and remade when that fails.
#
# Sizing from the laptop pilot (2026-10-13, 30 queries, 101 alphabet and k values, measured
# with /usr/bin/time -l): 227 s and a peak of 2.1 GB on one core. Index building is a fixed
# cost of about 2 s per alphabet and k; the rest grows with the number of queries, so 300
# queries take about 25 min on a laptop core. 2 h covers a Sherlock core up to 4x slower;
# 8 G is four times the measured peak. Read the real peak after the run with
# `sacct -j <jobid> -o JobID,MaxRSS,Elapsed`.
#
# Copy the table back to the laptop worktree (about 2 MB):
#
#   rsync -av sherlock-dtn:/scratch/users/olgabot/2024-kmerseek-analysis-fig1/tables/275_seed_frontier_hits_per_query.tsv \
#       /Users/olga/code/2024-kmerseek-analysis-fig1/tables/
set -euo pipefail

: "${SCRATCH:?}" "${SLURM_SUBMIT_DIR:?}"
cd "$SLURM_SUBMIT_DIR"
export NB275_PROTEOME=${NB275_PROTEOME:-$SCRATCH/data/gencode/human/v49/gencode.v49.pc_translations.canonical.fa}
export NB275_N_QUERIES=${NB275_N_QUERIES:-300}
export NB275_OUT=${NB275_OUT:-$SLURM_SUBMIT_DIR/tables/275_seed_frontier_hits_per_query.tsv}
module load python/3.12.1
VENV=$SCRATCH/venvs/nb275
if ! "$VENV/bin/python" -c 'import numpy, polars' 2>/dev/null; then
    rm -rf "$VENV"
    python3 -m venv "$VENV"
    "$VENV/bin/pip" install --quiet --upgrade pip
    "$VENV/bin/pip" install --quiet --only-binary=:all: polars==1.34.0 'numpy<2.4'
    "$VENV/bin/python" -c 'import numpy, polars; print("numpy", numpy.__version__, "polars", polars.__version__)'
fi
"$VENV/bin/python" scripts/run_275_seed_frontier.py
