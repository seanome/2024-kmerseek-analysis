#!/bin/bash
#SBATCH --job-name=299-finish
#SBATCH --account=ayeletv
#SBATCH --partition=hns
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=299-finish-%j.out
#
# Notebook 299, last job (submitted by 299_gate.sbatch after every search task has ended):
# recall and shuffled-hit rate per setting, verdict.json, the figure, and step 5's Logan
# arithmetic. A setting whose search did not finish is left out and named in this log.
# Sizing: reads DIAMOND's hits.tsv (2_289 queries, at most a few million rows) and the
# reduced per-setting tables (one row per query, plus DIAMOND-matched pairs); 16 G is
# several times what polars needs for that.
set -euo pipefail

REPO=${SLURM_SUBMIT_DIR:?submit from the repo root}
WORK=${OMRGC_WORK:-$SCRATCH/299-omrgc-pilot}
E=$WORK/env/bin
cd "$REPO"
"$E/python" notebooks/299_omrgc_sketch_pilot.py score
"$E/python" notebooks/299_omrgc_sketch_pilot.py logan
echo "Done. Results: notebooks/299_data/verdict.json, tables/299_kill_test_*.csv, figures/299_recall_vs_*" \
    | tee -a "$WORK/job_ids.txt"
