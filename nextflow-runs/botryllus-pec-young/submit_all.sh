#!/bin/bash
# Submit everything for the young-colony PEC run in one go, so nothing waits on a person:
#   1. splash_prep.sbatch, 52-task array: one read file per library for SPLASH
#   2. splash.sbatch, one job per time window, each after all of 1 succeeds
#   3. the Nextflow head (scripts/nf-head): nf-core/rnaseq, then nf-core/differentialabundance
# 1-2 and 3 do not depend on each other. Job ids go to $OUT/job_ids.txt.
# Called by `make run`; the Makefile sets the variables below.
set -euo pipefail
: "${REPO:?}" "${OUT:?}" "${NEXTFLOW:?}"
WINDOWS="young_2h young_24h young_13_17d young_21_28d young_49_52d young_104_132d"

mkdir -p "$OUT/logs"
ids=$OUT/job_ids.txt
# Sherlock's git is 1.8.3, which has no `git -C`.
echo "# submitted $(date '+%F %T') from $(cd "$REPO" && git rev-parse --short HEAD)" >> "$ids"

prep=$(sbatch --parsable --chdir "$OUT" "$REPO/splash_prep.sbatch" "$REPO")
echo "splash_prep $prep" | tee -a "$ids"
for w in $WINDOWS; do
  j=$(sbatch --parsable --chdir "$OUT" --job-name "pec-young-splash-$w" \
        --dependency "afterok:$prep" --kill-on-invalid-dep=yes \
        "$REPO/splash.sbatch" "$REPO" "$w")
  echo "splash_$w $j" | tee -a "$ids"
done

cd "$OUT"
NF_HEAD_DETACH=1 "$REPO/../../scripts/nf-head" "$REPO/run_pipelines.sh" "$REPO" "$OUT" "$NEXTFLOW"
echo "nf_head $(cat "$OUT/.nf-head.jobid")" | tee -a "$ids"
