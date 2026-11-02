#!/bin/bash
#SBATCH --job-name=nb361-sra-sketch
#SBATCH --account=ayeletv
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/%x.%A_%a.out
# One run of runs.tsv (array index = row, 0-based, header excluded; or RUNS_FILE lists runs).
# 1. pec-tools best-frame: per read, among frames without a stop codon and not of low complexity (FRAME_RULE=orpheum,
#    orpheum's rule), keep the one with the most Swiss-Prot k-mers (alphabet FRAME_ALPHA, k FRAME_K;
#    orpheum's defaults protein k7) and write it when its matching share >= each cutoff.
#    CUTOFFS must be written as Rust prints them (0.1, 0.5), because they name the files.
# 2. branchwater manysketch: one signature per run per cutoff, at the six search settings.
# Memory and time are set at submission from the run size.
set -euo pipefail
: "${PEC_ROOT:?}" "${PEC_CODE:?}" "${CUTOFFS:?}"
export PATH=$PEC_ROOT/env/bin:$PATH
FRAME_ALPHA=${FRAME_ALPHA:-protein}
FRAME_K=${FRAME_K:-7}
FRAME_RULE=${FRAME_RULE:-orpheum}
W=$PEC_ROOT/sra_search
cd "$W"
if [ -n "${RUNS_FILE:-}" ]; then
  run=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$RUNS_FILE")
else
  run=$(awk -F'\t' 'NR>1{print $1}' runs.tsv | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")
fi
files=$(awk -F'\t' -v r="$run" '$1==r{print $9}' runs.tsv | tr ';' ' ')
for f in $files; do
  [ -s "$f" ] || { echo "missing reads $f for $run"; exit 1; }
  case $f in "$W"/reads/*) [ -e "$f.done" ] || { echo "download of $f not verified"; exit 1; };; esac
done
O=$W/runs/$run
mkdir -p "$O"
[ -e "$O/sketch.done" ] && { echo "$run done"; exit 0; }
L=${L_SCRATCH:-/tmp}/nb361_$run
mkdir -p "$L"
trap 'command rm -rf "$L"' EXIT
PT=$PEC_CODE/pec-tools/target/release/pec-tools

/usr/bin/time -v -o "$O/best_frame.time" bash -c "set -o pipefail; zcat $files | $PT best-frame $PEC_ROOT/db/swissprot.faa $FRAME_ALPHA $FRAME_K $L/$run $CUTOFFS $FRAME_RULE 6" \
  2> "$O/best_frame.log"
command cp "$L/$run.hist.tsv" "$L/$run.summary.tsv" "$O/"
for c in ${CUTOFFS//,/ }; do
  faa=$L/$run.ge$c.faa
  wc -c < "$faa" > "$O/ge$c.faa_bytes"
  printf 'name,genome_filename,protein_filename\n%s,,%s\n' "$run" "$faa" > "$L/ms.$c.csv"
  /usr/bin/time -v -o "$O/manysketch.ge$c.time" sourmash scripts manysketch "$L/ms.$c.csv" -c 1 \
    -p hp,k=27,scaled=10 -p hp,k=30,scaled=10 -p dayhoff,k=12,scaled=10 -p dayhoff,k=13,scaled=10 \
    -p protein,k=6,scaled=10 -p protein,k=7,scaled=10 -o "$O/$run.ge$c.sig.zip" > "$O/manysketch.ge$c.log" 2>&1
done
touch "$O/sketch.done"
