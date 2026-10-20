#!/bin/bash
# Notebook 288 on Sherlock: submit both jobs at once. The heavy job waits for the pairwise
# job and starts only if it succeeded (afterok); if the pairwise job fails, Slurm cancels
# the heavy one instead of leaving it pending forever.
#
# Run from a Sherlock login node, in a clone of olgabot/two-step-search-288:
#   cd $SCRATCH/288-two-step-repo && bash notebooks/288_submit.sh
#
# DRY=1 prints the two sbatch commands without submitting.
#
# Asks, from notebook 241's runs.jsonl (see the sizing notes in 288_two_step_search.sbatch):
#   pairwise  210 indexes of 1-14 proteins + 210 searches of 202 queries: 8 CPU, 16 GB, 1 h
#   heavy     202 queries x 70 settings on notebook 283's human indexes (built here if those
#             are gone) + step 2: 16 CPU, 128 GB, 10 h (about 2.5 h expected)
set -euo pipefail

cd "$(dirname "$0")/.."
SB=notebooks/288_two_step_search.sbatch
PAIRWISE=(--job-name=288-pairwise --export=ALL,STEP=pairwise -c 8 --mem=16G --time=01:00:00)
HEAVY=(--job-name=288-heavy --export=ALL,STEP=heavy -c 16 --mem=128G --time=10:00:00)

if [ "${DRY:-0}" = 1 ]; then
    echo "sbatch --parsable ${PAIRWISE[*]} $SB"
    echo "sbatch --parsable --dependency=afterok:<pairwise job id> --kill-on-invalid-dep=yes ${HEAVY[*]} $SB"
    exit 0
fi

# Two copies of either job would write the same output folder.
queued=$(squeue -h -u "$USER" -o "%j" | grep -E '^288-(pairwise|heavy)$' || true)
if [ -n "$queued" ]; then
    echo "288: already in the queue, not submitting again:" >&2
    squeue -u "$USER" -o "%i %j %t %M %R" | grep -E 'JOBID|288-' >&2
    exit 1
fi

first=$(sbatch --parsable "${PAIRWISE[@]}" "$SB")
first=${first%%;*} # --parsable prints "id" or "id;cluster"
second=$(sbatch --parsable --dependency=afterok:"$first" --kill-on-invalid-dep=yes "${HEAVY[@]}" "$SB")
second=${second%%;*}
echo "288: pairwise job $first, heavy job $second (starts after $first succeeds)"
echo "288: logs in $PWD/288-two-step-<job id>.out"
