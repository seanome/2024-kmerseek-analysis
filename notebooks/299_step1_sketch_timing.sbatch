#!/bin/bash
#SBATCH --job-name=299-step1
#SBATCH --partition=hns
#SBATCH --array=0-26
#SBATCH --cpus-per-task=1
#SBATCH --mem=40G
#SBATCH --time=23:00:00
#SBATCH --output=299-step1-%A_%a.out
#
# Notebook 299, step 1: time `sourmash sketch translate` on one CPU on the 1 GB slice, one
# array task per setting (protein, dayhoff, hp x k 7, 10, 13 x scaled 10, 100, 1000).
# Submitted by notebooks/299_submit_all.sh to start when 299_setup.sbatch succeeds.
#
# Sizing: the rule's budget is 1_536 CPU-hours (two days of a 32-CPU node) for all nine
# settings of one scaled value over the whole catalog (size read from catalog_stats.json
# after setup). If the catalog is near 40 GB, that allows about 4 CPU-hours per GB per
# setting. `timeout` stops a setting at 40_000 s on the 1 GB slice: nine settings at that
# rate exceed 1_536 CPU-hours for any catalog over 15 GB (the gzipped file alone is
# 29.6 GB), so step1-table keeps 40_000 s as a lower bound. The .zip step has the same limit, so
# 23 h covers both plus start-up. A laptop rate of about 2.5-3.5 CPU-s per MB
# (catalog's first 2-8 MB, start-up included) suggests about 1 h per task.
# 40 G covers scaled 10, whose earlier 8 G attempts were killed (array 48668306).
set -euo pipefail

REPO=${SLURM_SUBMIT_DIR:?submit from the repo root}
WORK=${OMRGC_WORK:-$SCRATCH/299-omrgc-pilot}
E=${OMRGC_ENV_PREFIX:-${GROUP_HOME:?}/conda/envs/omrgc299}/bin
LIMIT=$("$E/python" -c "import json; print(json.load(open('$WORK/catalog_stats.json'))['step1_time_limit_s'])")
read -r ALPHABET K SCALED < <("$E/python" "$REPO/notebooks/299_omrgc_sketch_pilot.py" setting "$SLURM_ARRAY_TASK_ID")
SID=$ALPHABET.k$K.s$SCALED
echo "host $(hostname); $(lscpu | grep 'Model name'); setting $SID"

# Two timed parts, both counted: sketching to .sig.gz, then writing the .zip index of the
# non-empty sketches (drop-empty). Writing .zip directly was the first run's mistake: the
# .zip writer slows down on repeated sketches, and all empty sketches are the same, so all
# nine scaled 1000 settings and hp k7 scaled 100 hit the time limit (jobs 48551687).
rm -f "${WORK:?}/step1/${SID:?}".*
set +e
/usr/bin/time -v -o "$WORK/step1/$SID.time" timeout "$LIMIT" \
    "$E/sourmash" sketch translate --singleton -p "$ALPHABET,k=$K,scaled=$SCALED,noabund" \
    -o "$WORK/step1/$SID.sig.gz" "$WORK/slice/OM-RGC_v2_genes_first_1GB.fna"
STATUS=$?
set -e
echo "sketch exit $STATUS"
tail -n 25 "$WORK/step1/$SID.time"
# 124 is `timeout` stopping it: a lower bound, read as such by step1-table.
# The .zip step gets its own time limit: if it stops, step1-table keeps the limit as a
# lower bound instead of the task failing and cancelling the gate.
ZIP_STATUS=0
if [ "$STATUS" = 0 ]; then
    set +e
    /usr/bin/time -v -o "$WORK/step1/$SID.zip.time" timeout "$LIMIT" "$E/python" \
        "$REPO/notebooks/299_omrgc_sketch_pilot.py" drop-empty "$WORK/step1/$SID.sig.gz" "$WORK/step1/$SID.sig.zip"
    ZIP_STATUS=$?
    set -e
    echo "zip exit $ZIP_STATUS"
fi
{ [ "$STATUS" = 0 ] || [ "$STATUS" = 124 ]; } && { [ "$ZIP_STATUS" = 0 ] || [ "$ZIP_STATUS" = 124 ]; }
