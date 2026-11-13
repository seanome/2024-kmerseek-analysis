#!/bin/bash
# Notebook 286: resubmit the step 2 tasks that failed or are still waiting, with time and
# memory from the first step 2 run (2026-10-20/21), then the score job after all of them.
# Run on a Sherlock login node from a checkout of olgabot/intermediate-sequence-286:
#
#   cd /scratch/users/olgabot/286-repo && git pull && bash notebooks/286_resubmit_step2.sh
#
# A step 2 task resumes: finished chunks are kept, and a setting whose table is already
# written exits at once. So resubmitting a task costs only its missing chunks.
#
# Measured (sacct and step2/<setting>.timing.jsonl, 2026-10-21 06:45, saved as
# step2_sacct_2026-10-21.csv and step2_remaining_2026-10-21.csv in $NB286_DIR):
#   100 of 152 settings finished. Of the 41 that asked 496 GB, the largest used 63 GB
#   (median 10 GB); 43 more still wait in array 48704089 for 496 GB nodes.
#   Failed, with what is left (4 processes per task unless noted):
#     task 1  dayhoff6.k9   OUT_OF_MEMORY at 48 GB after 39 min; 5 chunks of 75_000
#                           residues left, 1 of them holds Ced9/P66 intermediates.
#                           Ask 4x the killed ask: 192 GB, 2 h.
#     task 17 gbmr4.k10     TIMEOUT at 6 h; 139 chunks, 824_284 residues left at
#                           0.0075 s per residue: 0.43 h of 4 processes. 160 GB held
#                           through 1_818 chunks (largest process 43 GB).
#     task 18 gbmr4.k13     TIMEOUT at 2 h; 283 chunks, 1_679_920 residues at 0.0029 s:
#                           0.33 h. Largest process 5 GB.
#                           Tasks 1, 17 and 18 go in one array at 192 GB, 2 h.
#     task 25 gbmr7.k8      TIMEOUT at 12 h; 286 chunks, 1_697_758 residues at 0.0164 s:
#                           1.9 h, x1.5 = 2.9 h. Largest process 94 GB; 288 GB held. 288 GB, 4 h.
#   Still waiting, by seed bits:
#     below 21 bits: the 21 finished settings below 21 bits took up to 3.9 h (gbmr7.k10,
#     predicted 3.6 h) and 108 GB; dayhoff6.k9 at 20.5 bits ran out at 48 GB. The 7
#     waiting ones are predicted at 1.8 h or less. 192 GB, 4 h.
#     21 bits or more: the 79 finished settings at 21+ bits took at most 0.62 h; the
#     waiting alphabets used at most 63 GB at any k (gbmr7.k12 used 107 GB but is not
#     waiting). 96 GB, 2 h.
#   Score: as before, 16 CPUs, 64 GB, 4 h, after every step 2 task; it now refuses to
#   score while any setting is unfinished (ALLOW_PARTIAL=1 scores early into results_partial/).
set -euo pipefail

: "${SCRATCH:?}"
cd "$(dirname "$0")"
export NB286_DIR=${NB286_DIR:-$SCRATCH/286-intermediate-sequence}
OLD_ARRAY=48704089
OLD_SCORE=48704093
# Settings below 21 seed bits among those waiting on 2026-10-21 (from step2_plan.csv).
LOW_BITS=" 91 99 107 114 123 130 145 "

pending=$( (squeue -r -h -j "$OLD_ARRAY" -t PD -o %K 2>/dev/null || true) | sort -n | tr '\n' ' ')
echo "waiting tasks of $OLD_ARRAY: ${pending:-none}"
for t in $pending; do scancel "${OLD_ARRAY}_${t}"; done
scancel "$OLD_SCORE" 2>/dev/null || true

low=""; high=""
for t in $pending; do
    case "$LOW_BITS" in *" $t "*) low="$low,$t" ;; *) high="$high,$t" ;; esac
done

sub() {  # sub <array list> <hours> <mem GB>
    sbatch --parsable --array="$1" --time="$2:00:00" --mem="$3G" --cpus-per-task=32 \
        --export=ALL,NB286_DIR="$NB286_DIR",NB286_WORKERS=4 286_step2.sbatch
}
ids="$OLD_ARRAY"
ids="$ids:$(sub 1,17,18 2 192)"
ids="$ids:$(sub 25 4 288)"
[ -n "$low" ] && ids="$ids:$(sub "${low#,}" 4 192)"
[ -n "$high" ] && ids="$ids:$(sub "${high#,}" 2 96)"
score=$(sbatch --parsable --dependency=afterany:"$ids" --export=ALL,NB286_DIR="$NB286_DIR" 286_score.sbatch)
echo "step 2 jobs: $ids"
echo "score job: $score (after all of them)"
echo "watch: squeue -u \$USER -o \"%i %j %t %M %l %R %E\" | grep 286"
