#!/bin/bash
# One array task of notebook 290: row SLURM_ARRAY_TASK_ID of the list file $LIST (a slice
# of plan/tasks.tsv written by submit.py). Submitted by `make pilot` / `make full`, never by
# hand: submit.py sets memory, time and how many run at once from plan/resources.tsv.
#
#   kind dipeptide | window10   search the shuffled chunk
#   kind real                   search the real chunk again (positive control)
#   kind realfile               read the run's own region file (no search)
#
# Indexes and the run's chunks and region files are mounted read-only; only $BASE is
# writable. A task whose .check.tsv exists is skipped, so a resubmitted list redoes only
# what did not finish.
#SBATCH -A ayeletv
#SBATCH -p hns,normal
#SBATCH -c 8
set -euo pipefail
: "${LIST:?}" "${BASE:?}" "${STAGE:?}" "${IMG:?}"
IFS=$'\t' read -r species clade chunk kind alphabet k pen xdrop index run_regions \
    < <(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$LIST")
[ -n "$kind" ] || { echo "290: no row $SLURM_ARRAY_TASK_ID in $LIST" >&2; exit 2; }
slug="$chunk.$kind.$alphabet.k$k.s1.lcfalse.extend-c$pen"
out="$BASE/out/$species"
mkdir -p "$out"
prefix="$out/$slug"
[ -s "$prefix.check.tsv" ] && { echo "290: $slug done already"; exit 0; }
echo "290: $species $slug on $(hostname), $(date)"

if [ "$kind" = realfile ]; then
    [ -s "$run_regions" ] || { echo "290: missing $run_regions" >&2; exit 2; }
    zstd -dc "$run_regions" \
      | apptainer exec --no-home -B "$BASE" -B "$STAGE:$STAGE:ro" "$IMG" \
            python3 "$STAGE/summarize_regions.py" "$prefix" "$slug"
    exit 0
fi

# The query: the run's chunk (md5 from plan/chunks.tsv) or its shuffled copy (md5 from
# chunks/shuffles.tsv). Refuse anything that is not the recorded file.
if [ "$kind" = real ]; then
    query=$(awk -F'\t' -v s="$species" -v c="$chunk" '$1==s && $3==c {print $4}' "$BASE/plan/chunks.tsv")
    md5=$(awk -F'\t' -v s="$species" -v c="$chunk" '$1==s && $3==c {print $5}' "$BASE/plan/chunks.tsv")
else
    query=$(awk -F'\t' -v s="$species" -v c="$chunk" -v k="$kind" '$1==s && $2==c && $3==k {print $4}' "$BASE/chunks/shuffles.tsv")
    md5=$(awk -F'\t' -v s="$species" -v c="$chunk" -v k="$kind" '$1==s && $2==c && $3==k {print $5}' "$BASE/chunks/shuffles.tsv")
fi
[ -n "$query" ] && [ "$(md5sum < "$query" | cut -d' ' -f1)" = "$md5" ] \
    || { echo "290: query $query is not the recorded file (md5 $md5)" >&2; exit 2; }
[ -d "$index" ] || { echo "290: no index $index; not building one" >&2; exit 2; }

apptainer exec --no-home \
    -B "$BASE" \
    -B "$STAGE:$STAGE:ro" \
    -B "$(dirname "$query"):$(dirname "$query"):ro" \
    -B "$index:$index:ro" \
    "$IMG" bash "$STAGE/run_one.sh" "$kind" "$alphabet" "$k" "$pen" "$xdrop" "$query" "$index" \
        "$prefix" "$BASE/plan/awk_filter.txt"
echo "290: done $(date)"
