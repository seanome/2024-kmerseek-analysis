#!/bin/bash
# One kmerseek search of notebook 290, run INSIDE the dark-set run's image
# (olgabot/kmerseek:0.4.0-rc5), so kmerseek, awk (mawk) and zstd are the run's own.
#
#   run_one.sh <kind> <alphabet> <k> <penalty> <xdrop> <query.fasta> <index> <out_prefix> <awk_filter_file>
#
# <kind> real keeps the whole region file (<out_prefix>.regions.csv.zst, the positive
# control, compared with the run's file) and then reads it; a shuffled kind is read as it
# streams and only the summaries are kept. The kmerseek command is the run's
# (tasks.py checked it against all 294 .command.sh files), and the E <= 10_000 awk filter is
# the run's own text, read from <awk_filter_file>.
set -euo pipefail
kind=$1 alphabet=$2 k=$3 pen=$4 xdrop=$5 query=$6 index=$7 prefix=$8 awkfile=$9
here=$(cd "$(dirname "$0")" && pwd)
awk_filter=$(cat "$awkfile")
label=$(basename "$prefix")

search() {
    kmerseek search \
        --alphabet "$alphabet" --ksize "$k" \
        --query "$query" --target "$index" \
        --extend-mismatch-penalty "$pen" --extend-xdrop "$xdrop" --ka-queries 0 \
        --chain-max-gap 30 --chain-max-shift 10 \
        --threshold 0.0 \
        --min-shared-kmers 2 \
        --max-query-pvalue 0.05 \
        --min-region-score 1.3 \
        2> "$prefix.log"
}

set +e
if [ "$kind" = real ]; then
    search | eval "$awk_filter" | zstd -T2 -f -o "$prefix.regions.csv.zst.tmp"
else
    search | eval "$awk_filter" | python3 "$here/summarize_regions.py" "$prefix" "$label"
fi
status=("${PIPESTATUS[@]}")
set -e
if [ "${status[0]}" -ne 0 ] || [ "${status[1]}" -ne 0 ] || [ "${status[2]}" -ne 0 ]; then
    echo "290: pipeline failed, exit codes ${status[*]}; kmerseek log:" >&2
    tail -5 "$prefix.log" >&2
    exit 1
fi
if [ "$kind" = real ]; then
    mv "$prefix.regions.csv.zst.tmp" "$prefix.regions.csv.zst"
    zstd -dc "$prefix.regions.csv.zst" | python3 "$here/summarize_regions.py" "$prefix" "$label"
fi
