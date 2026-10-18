#!/bin/bash
#SBATCH --job-name=299-step1
#SBATCH --account=ayeletv
#SBATCH --partition=hns
#SBATCH --array=0-17
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=299-step1-%A_%a.out
#
# Notebook 299, step 1: time `sourmash sketch translate` on one CPU on the 1 GB slice, one
# array task per setting (protein, dayhoff, hp x k 7, 10, 13 x scaled 100, 1000). After
# 299_setup.sbatch has finished:
#
#   cd $SCRATCH/2024-kmerseek-analysis-299 && sbatch notebooks/299_step1_sketch_timing.sbatch
#
# Sizing: the rule's budget is 48 CPU-hours for all nine settings of one scaled value over
# the whole catalog (size read from catalog_stats.json after setup). If the catalog is near
# 40 GB, that allows about 8 CPU-minutes per GB per setting; two days of a 32-CPU node would
# allow about 4 CPU-hours. `timeout` stops a setting at 40_000 s on the 1 GB slice: nine
# settings at that rate exceed 48 CPU-hours for any catalog over 0.5 GB and 1_536 over
# 15 GB (the gzipped file alone is 29.6 GB), so step1-table keeps 40_000 s as a lower
# bound. 12 h covers that plus start-up. A laptop rate of about 2.5-3.5 CPU-s per MB
# (catalog's first 2-8 MB, start-up included) suggests about 1 h per task. 8 G: a
# --singleton sketch holds only the current record's hashes plus the output.
set -euo pipefail

REPO=${SLURM_SUBMIT_DIR:?submit from the repo root}
WORK=${OMRGC_WORK:-$SCRATCH/299-omrgc-pilot}
E=$WORK/env/bin
LIMIT=$("$E/python" -c "import json; print(json.load(open('$WORK/catalog_stats.json'))['step1_time_limit_s'])")
read -r ALPHABET K SCALED < <("$E/python" "$REPO/notebooks/299_omrgc_sketch_pilot.py" setting "$SLURM_ARRAY_TASK_ID")
SID=$ALPHABET.k$K.s$SCALED
echo "host $(hostname); $(lscpu | grep 'Model name'); setting $SID"

set +e
/usr/bin/time -v -o "$WORK/step1/$SID.time" timeout "$LIMIT" \
    "$E/sourmash" sketch translate --singleton -p "$ALPHABET,k=$K,scaled=$SCALED,noabund" \
    -o "$WORK/step1/$SID.sig.zip" "$WORK/slice/OM-RGC_v2_genes_first_1GB.fna"
STATUS=$?
set -e
echo "exit $STATUS"
tail -n 25 "$WORK/step1/$SID.time"
# 124 is `timeout` stopping it: a lower bound, read as such by step1-table.
[ "$STATUS" = 0 ] || [ "$STATUS" = 124 ]
