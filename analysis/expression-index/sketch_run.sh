#!/bin/bash
# Download one run's FASTQ files from ENA, sketch them, check the sketch, delete the FASTQs.
#   sketch_run.sh ROW_NUMBER RUNS_TSV WORK_DIR
# ROW_NUMBER is 1-based over the data rows of RUNS_TSV (sorted smallest first).
# Writes WORK_DIR/sketches/<accession>.sig.gz, WORK_DIR/index_log/<accession>.tsv and
# WORK_DIR/downloads/<accession>.tsv (file, bytes, seconds, or the curl error).
set -uo pipefail
row=$1; runs=$2; work=$3
here=$(cd "$(dirname "$0")" && pwd)
bin=$here/indexer/target/release/expression-indexer
read -r acc u1 u2 u0 < <(python3 - "$runs" "$row" <<'EOF'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1]), delimiter="\t"))
r = rows[int(sys.argv[2]) - 1]
print(r["accession"], r.get("ena_fastq_http_1") or "-", r.get("ena_fastq_http_2") or "-", r.get("ena_fastq_http") or "-")
EOF
)
mkdir -p "$work"/{fastq,sketches,index_log,downloads}
if [ -s "$work/sketches/$acc.sig.gz" ] && [ -s "$work/index_log/$acc.tsv" ]; then
    echo "$acc already sketched"; exit 0
fi
urls=()
if [ "$u1" != "-" ]; then urls+=("$u1"); [ "$u2" != "-" ] && urls+=("$u2"); elif [ "$u0" != "-" ]; then urls+=("$u0"); fi
if [ ${#urls[@]} -eq 0 ]; then
    printf '%s\t-\t-\t-\tno ENA FASTQ link\n' "$acc" > "$work/downloads/$acc.tsv"; exit 1
fi
: > "$work/downloads/$acc.tsv"
files=()
for u in "${urls[@]}"; do
    case $u in http*) ;; *) u="https://$u";; esac
    f=$work/fastq/$(basename "$u")
    t=$(date +%s); ok=0
    for attempt in 1 2 3 4 5; do
        # resume a partial file; give up on a transfer that stalls below 1 byte/s for 60 s
        if err=$(curl -sS -L -C - --retry 10 --retry-all-errors --speed-limit 1 --speed-time 60 -o "$f" "$u" 2>&1); then ok=1; break; fi
        sleep $((attempt * 30))
    done
    if [ $ok -ne 1 ] || ! gzip -t "$f" 2>/dev/null; then
        printf '%s\t%s\t-\t%s\t%s\n' "$acc" "$u" $(( $(date +%s) - t )) "download failed: ${err:-gzip -t failed}" >> "$work/downloads/$acc.tsv"
        command rm -f "$f"; exit 1
    fi
    printf '%s\t%s\t%s\t%s\tok\n' "$acc" "$u" "$(stat -c %s "$f")" $(( $(date +%s) - t )) >> "$work/downloads/$acc.tsv"
    files+=("$f")
done
"$bin" sketch "$acc" "$here/k_for_protein7_bits.tsv" "$work/sketches" "${files[@]}" > "$work/index_log/$acc.tsv.tmp" || exit 1
# check the sketch before deleting the reads: it must be valid gzip and hold one signature per alphabet
n_alpha=$(grep -vc -e '^#' -e '^alphabet' "$here/k_for_protein7_bits.tsv")
n_sig=$(gzip -dc "$work/sketches/$acc.sig.gz" | grep -o '"molecule"' | wc -l)
if [ "$n_sig" -ne "$n_alpha" ]; then echo "$acc: $n_sig signatures, expected $n_alpha"; exit 1; fi
mv "$work/index_log/$acc.tsv.tmp" "$work/index_log/$acc.tsv"
command rm -f "${files[@]}"
echo "$acc done"
