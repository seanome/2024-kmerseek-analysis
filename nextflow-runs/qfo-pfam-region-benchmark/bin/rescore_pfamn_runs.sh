#!/usr/bin/env bash
# Rescore the Pfam-N metrics of several finished runs, one after another, with
# rescore_pfamn_from_calls.py. Called by `make rescore-pfamn-from-calls` inside the pipeline
# image; see that target for what it writes and why.
#
#   rescore_pfamn_runs.sh DATA_DIR JOBS SOURCE_COMMIT RUN [RUN ...]
#
# A run whose pfamn_rescored_metrics.parquet already exists is skipped. A run that stopped
# part way resumes from the per-species parts it saved (pfamn_rescore_parts/).
set -euo pipefail

data=$1 jobs=$2 commit=$3
shift 3
export POLARS_MAX_THREADS=1
here=$(cd "$(dirname "$0")" && pwd)

for r in "$@"; do
    d=$data/$r/results
    if [ -s "$d/pfamn_rescored_metrics.parquet" ]; then
        echo "== $r: $d/pfamn_rescored_metrics.parquet exists, skipped"
        continue
    fi
    # midi and midi-plus kept target disorder in their --db_cache (../results); mini has
    # no --db_cache, so its own outdir holds it.
    dis=$data/../results/disorder
    [ -d "$d/disorder" ] && dis=$d/disorder
    echo "== $r (target disorder from $dis) $(date '+%F %T')"
    python3 "$here/rescore_pfamn_from_calls.py" \
        --run-dir       "$d" \
        --annotations   "$data/$r/annotations" \
        --pfamn-dir     "$data/pfamn" \
        --disorder-dir  "$dis" \
        --jobs          "$jobs" \
        --source-commit "$commit"
done
echo "== all done $(date '+%F %T')"
