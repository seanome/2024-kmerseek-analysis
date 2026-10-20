#!/bin/bash
#SBATCH --job-name=299-sketch
#SBATCH --account=ayeletv
#SBATCH --partition=hns
#SBATCH --cpus-per-task=1
#SBATCH --output=299-sketch-%A_%a.out
#
# Notebook 299, step 3a: sketch the whole catalog with `sourmash sketch translate`, all nine
# alphabet and k settings of one scaled value in one pass, one array task per group of
# catalog parts, plus 32 tasks for the parts of the 1% random gene sample. Submitted by
# 299_gate.sbatch, which sets the number of tasks, time and memory from step 1's measured
# rates (plan-step3) and passes N_CHUNKS and SCALED in the environment.
set -euo pipefail

REPO=${SLURM_SUBMIT_DIR:?submit from the repo root}
WORK=${OMRGC_WORK:-$SCRATCH/299-omrgc-pilot}
E=$WORK/env/bin
: "${N_CHUNKS:?}" "${SCALED:?}"
I=$SLURM_ARRAY_TASK_ID
echo "host $(hostname); $(lscpu | grep 'Model name'); chunk $I of $N_CHUNKS, scaled $SCALED"

PARAMS=()
for A in protein dayhoff hp; do for K in 7 10 13; do PARAMS+=(-p "$A,k=$K,scaled=$SCALED,noabund"); done; done
OUT=$WORK/catalog_sketch/s$SCALED
mkdir -p "$OUT" "$OUT/sample"
if (( I < N_CHUNKS )); then
    # Tasks 0..N_CHUNKS-1: the catalog parts whose index mod N_CHUNKS is this task's.
    mapfile -t ALL < <(ls "$WORK/parts"/*.fna* | sort)
    MINE=()
    for idx in "${!ALL[@]}"; do (( idx % N_CHUNKS == I )) && MINE+=("${ALL[$idx]}"); done
    NAME=$OUT/chunk$I
else
    # Tasks N_CHUNKS..N_CHUNKS+31: one part each of the 1% random gene sample.
    mapfile -t ALL < <(ls "$WORK/sample/parts"/*.fna* | sort)
    MINE=("${ALL[$((I - N_CHUNKS))]}")
    NAME=$OUT/sample/part$((I - N_CHUNKS))
fi
# Sketch to .sig.gz, then keep only non-empty sketches in the .zip that manysearch reads
# (see drop-empty in 299_omrgc_sketch_pilot.py for why).
/usr/bin/time -v -o "$NAME.time" \
    "$E/sourmash" sketch translate --singleton "${PARAMS[@]}" -o "$NAME.sig.gz" "${MINE[@]}"
/usr/bin/time -v -o "$NAME.zip.time" \
    "$E/python" "$REPO/notebooks/299_omrgc_sketch_pilot.py" drop-empty "$NAME.sig.gz" "$NAME.sig.zip"
rm -f "${NAME:?}.sig.gz"
tail -n 25 "$NAME.time" "$NAME.zip.time"
