#!/bin/bash
#SBATCH --job-name=287-shuf-chains
#SBATCH --account=ayeletv
#SBATCH --partition=hns
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=287-shuf-chains-%j.out
#
# Notebook 287's null: chain scores for the 63 queries of notebook 284's shuffled-query
# search (Ced9, P66, BHF and 20 dipeptide-shuffled copies of each). Needs 284's search
# step to have finished ($SCRATCH/284-shuffle-agreement/regions), which needs notebook
# 283's 152 rebuilt indexes. Submit from a checkout of olgabot/sum-statistics-287, after
# the 284 job:
#
#   git clone --branch olgabot/sum-statistics-287 https://github.com/seanome/2024-kmerseek-analysis $SCRATCH/287-sum-statistics-repo
#   cd $SCRATCH/287-sum-statistics-repo && sbatch --dependency=afterok:<284 job id> scripts/287_shuffle_chains.sbatch
#
# Sizing, from the laptop run of scripts/287_chain_scores.py (2026-10-13): Ced9, P66 and
# BHF, 7 variants each, took about 3 min including a one-off 1-2 min table build; the
# largest query (P66, 590_778 calls with an E-value) held well under 4 GB. Here 63 queries
# x 4 variants: about 20-40 min on Sherlock cores up to 2x slower. 16 GB is 4x the laptop's
# largest query. After the run: sacct -j <jobid> -o JobID,MaxRSS,Elapsed.
#
# Copy back the one small table:
#   pixi exec rsync -av sherlock-dtn:/scratch/users/olgabot/284-shuffle-agreement/287_shuffle_chains.csv \
#       /Users/olga/code/2024-kmerseek-analysis-287/notebooks/287_shuffle_chains.csv
set -euo pipefail

: "${SCRATCH:?}" "${SLURM_SUBMIT_DIR:?}"
export NB284_DIR=${NB284_DIR:-$SCRATCH/284-shuffle-agreement}
# notebook 283's venv (polars 1.34.0, numpy 2.x) on the python/3.12.1 module
PY=$HOME/venvs/nb283/bin/python
module load python/3.12.1
[ -s "$PY" ] || { echo "287: missing $PY (made by notebook 283's sbatch header)" >&2; exit 2; }

cd "$SLURM_SUBMIT_DIR"
$PY scripts/287_shuffle_chains.py
echo "287: done. Peak memory and time: sacct -j $SLURM_JOB_ID -o JobID,MaxRSS,Elapsed"
