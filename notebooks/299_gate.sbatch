#!/bin/bash
#SBATCH --job-name=299-gate
#SBATCH --account=ayeletv
#SBATCH --partition=hns
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=299-gate-%j.out
#
# Notebook 299: runs after every step 1 task has finished (submitted by 299_submit_all.sh).
# Reads step 1's verdict. If the catalog does not fit the budget, the test is dead and
# nothing more is submitted. Otherwise it submits the rest, each waiting on what it needs:
#
#   DIAMOND (step 2) ─────────────────┐
#   sketch catalog, scaled 100 ───────┼─► search (step 3, 18 tasks) ─► finish (score, Logan)
#   sketch catalog, scaled 1000 ──────┘
#
# The sketch arrays are sized here, from step 1's measured rates (plan-step3), because
# their size cannot be known before step 1. A job whose dependency fails is cancelled
# (--kill-on-invalid-dep), so a failure shows in sacct instead of a job pending forever.
# Sizing: step1-table and plan-step3 read a few small files; seconds, well under 1 G.
set -euo pipefail

REPO=${SLURM_SUBMIT_DIR:?submit from the repo root}
WORK=${OMRGC_WORK:-$SCRATCH/299-omrgc-pilot}
E=$WORK/env/bin
cd "$REPO"
SUB=(sbatch --parsable --chdir "$REPO" --kill-on-invalid-dep=yes)

"$E/python" notebooks/299_omrgc_sketch_pilot.py step1-table
"$E/python" notebooks/299_omrgc_sketch_pilot.py logan
# ONLY_SCALED (optional, e.g. 10): plan, sketch and search only that scaled value, and only
# if it fits the budget itself. AFTER_FINISH (optional): a job id the final finish also waits
# for, so the last scoring sees every setting.
FITS=$("$E/python" - "${ONLY_SCALED:-}" <<'PY'
import json, sys
v = json.load(open("notebooks/299_data/step1_verdict.json"))
only = sys.argv[1]
if only:
    print(any(r["scaled"] == int(only) and r["fits_two_compute_days"] for r in v["per_scaled"]))
else:
    print(v["fits_at_some_scaled"])
PY
)
if [ "$FITS" != "True" ]; then
    echo "Step 1 says dead${ONLY_SCALED:+ at scaled $ONLY_SCALED}: the catalog does not fit the budget. Nothing more submitted."
    echo "step 1 dead${ONLY_SCALED:+ at scaled $ONLY_SCALED}; nothing submitted" >> "$WORK/job_ids.txt"
    exit 0
fi

# DIAMOND writes diamond/done when it has finished everything; a resubmitted gate reuses it.
if [ -e "$WORK/diamond/done" ]; then
    DIAMOND=""
    echo "DIAMOND already done ($WORK/diamond/done); not resubmitted"
else
    DIAMOND=$("${SUB[@]}" notebooks/299_step2_diamond.sbatch)
    DIAMOND=${DIAMOND%%;*}
fi
SKETCH=()
while read -r SCALED N_CHUNKS LAST HOURS MEM; do
    ID=$("${SUB[@]}" --export=ALL,N_CHUNKS="$N_CHUNKS",SCALED="$SCALED" --array=0-"$LAST" \
        --time="$HOURS":00:00 --mem="$MEM"G notebooks/299_step3_sketch_catalog.sbatch)
    SKETCH+=("${ID%%;*}")
    echo "sketch scaled $SCALED: job ${ID%%;*}, $N_CHUNKS catalog chunks + 32 sample parts, $HOURS h, $MEM G each"
done < <("$E/python" notebooks/299_omrgc_sketch_pilot.py plan-step3 ${ONLY_SCALED:+--scaled "$ONLY_SCALED"})
[ "${#SKETCH[@]}" -ge 1 ] || { echo "plan-step3 gave no lines"; exit 1; }
DEPS=$(IFS=:; echo "${DIAMOND:+$DIAMOND:}${SKETCH[*]}")

if [ -n "${ONLY_SCALED:-}" ]; then
    ARRAY=$("$E/python" notebooks/299_omrgc_sketch_pilot.py indices --scaled "$ONLY_SCALED")
else
    ARRAY=$("$E/python" notebooks/299_omrgc_sketch_pilot.py indices --scaled 100 1000)
fi
# Scaled 10 sketches hold about 10 times the hashes of scaled 100, whose sample search peaked
# at 4.7 GB (array 48667936): 64 G instead of the job's 16 G.
MEMARG=()
if [ "${ONLY_SCALED:-}" = 10 ]; then MEMARG=(--mem=64G); fi
SEARCH=$("${SUB[@]}" "${MEMARG[@]}" --array="$ARRAY" --dependency=afterok:"$DEPS" notebooks/299_step3_search.sbatch)
SEARCH=${SEARCH%%;*}
# afterany: a setting whose search failed is left out of the score and named in the log.
FINISH=$("${SUB[@]}" --dependency=afterany:"$SEARCH${AFTER_FINISH:+:$AFTER_FINISH}" notebooks/299_finish.sbatch)
FINISH=${FINISH%%;*}
printf 'scaled %s: diamond %s; sketch %s; search %s (array %s); finish %s\n' \
    "${ONLY_SCALED:-100 1000}" "${DIAMOND:-reused}" "${SKETCH[*]}" "$SEARCH" "$ARRAY" "$FINISH" | tee -a "$WORK/job_ids.txt"
