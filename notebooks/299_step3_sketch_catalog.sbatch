#!/bin/bash
#SBATCH --job-name=299-sketch
#SBATCH --account=ayeletv
#SBATCH --partition=hns
#SBATCH --cpus-per-task=1
#SBATCH --output=299-sketch-%A_%a.out
#
# Notebook 299, step 3a: sketch the whole catalog with `sourmash sketch translate`, all nine
# alphabet and k settings of one scaled value in one pass, one array task per group of
# catalog parts, plus 32 tasks for the parts of the 1% random gene sample. Time, memory and the number of tasks come from step 1's measured rates:
# print the two commands (one per scaled value) with
#
#   $SCRATCH/299-omrgc-pilot/env/bin/python notebooks/299_omrgc_sketch_pilot.py plan-step3
#
# and submit what it prints. It refuses when step 1 killed the idea. Needs N_CHUNKS and
# SCALED in the environment (the printed command sets both).
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
/usr/bin/time -v -o "$NAME.time" \
    "$E/sourmash" sketch translate --singleton "${PARAMS[@]}" -o "$NAME.sig.zip" "${MINE[@]}"
tail -n 25 "$NAME.time"
