#!/bin/bash
# Refresh the accounting and resource table at every launch; resume only OOM jobs.
set -euo pipefail
: "${SCRATCH:?}"
BASE=${NB301_DIR:-$SCRATCH/301-short-feature-boundaries}
REPO=$PWD
module load python/3.12.1
# One folder per launch day, so each day's tables stay apart.
REC=$BASE/recovery-$(date +%Y%m%d)
mkdir -p "$REC"
sacct -S 2026-10-22 -X -P -n -o JobID,JobName%45,State,Elapsed,ReqMem,AllocCPUS > "$REC/accounting.txt"
python3 -I scripts/measure_301_recovery.py --logs "$BASE/step5" \
    --accounting "$REC/accounting.txt" --out "$REC/resources.tsv"
while IFS=$'\t' read -r alphabet oldjob state elapsed oldmem stage peak chunks mem hours basis; do
    [ "$state" = OUT_OF_MEMORY ] || continue
    # Do not submit a duplicate when this command is rerun.
    active=$(squeue -u "$USER" -h -o '%j' | grep -Fx "301-resume-$alphabet" || true)
    [ -z "$active" ] || { echo "already active: $alphabet"; continue; }
    job=$(sbatch --parsable -J "301-resume-$alphabet" --mem="${mem}G" --time="${hours}:00:00" \
        --export="ALL,ALPHABET=$alphabet,KMERSEEK=$SCRATCH/301-kmerseek-v0.4.0/target/release/kmerseek,CHUNK=100" \
        scripts/301_kmerseek_alphabet.sbatch)
    printf '%s\t%s\t%s\t%s\n' "$alphabet" "$job" "$mem" "$hours" | tee -a "$REC/submitted.tsv"
done < <(tail -n +2 "$REC/resources.tsv")
