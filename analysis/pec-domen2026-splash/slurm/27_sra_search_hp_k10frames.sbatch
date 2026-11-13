#!/bin/bash
#SBATCH --job-name=nb361-k10-search
#SBATCH --account=ayeletv
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x.%j.out
# Search the query proteins (one signature per protein) against every run signature made by
# 26_sra_k10_frame_sketch.sbatch, at HP k = 24, 27 and 30 only. Threshold 0 with -A: every
# query-run pair is reported, including zero overlap. Same queries as the earlier search
# (queries/queries.faa). Output: $PEC_ROOT/sra_search/search_hp_k10frames (search_all is not touched).
# File names keep the ms.ge<cutoff>.<alphabet>.k<k>.csv pattern that sra_collect.py reads; the
# cutoff is 0 (any matching 10-mer).
set -euo pipefail
: "${PEC_ROOT:?}"
export PATH=$PEC_ROOT/env/bin:$PATH
W=$PEC_ROOT/sra_search
QUERIES=${QUERIES:-queries/queries.faa}
O=$W/${OUT_DIR:-search_hp_k10frames}
mkdir -p "$O"
cd "$O"
printf 'name,genome_filename,protein_filename\nqueries,,%s\n' "$W/$QUERIES" > q.csv
/usr/bin/time -v -o queries.sketch.time sourmash scripts manysketch q.csv --singleton -c "$SLURM_CPUS_PER_TASK" \
  -p hp,k=24,scaled=10 -p hp,k=27,scaled=10 -p hp,k=30,scaled=10 -o queries.sig.zip > queries.sketch.log 2>&1
# hashes per signature, so an empty query or run sketch reads as "unsearchable", not "absent"
sourmash sig describe --csv queries.describe.csv queries.sig.zip > /dev/null 2>&1
ls "$W"/k10_frames/*/*.ge0.sig.zip > against.ge0.txt
echo "runs: $(wc -l < against.ge0.txt)"
sourmash sig describe --csv runs.ge0.describe.csv $(cat against.ge0.txt) > /dev/null 2>&1
for k in 24 27 30; do
  out=ms.ge0.hp.k$k.csv
  [ -s "$out" ] && continue
  /usr/bin/time -v -o "$out.time" sourmash scripts manysearch queries.sig.zip against.ge0.txt \
    -m hp -k "$k" -s 10 -t 0 -A --ignore-abundance -N -c "$SLURM_CPUS_PER_TASK" -o "$out.part" \
    > "$out.log" 2>&1
  mv "$out.part" "$out"
done
touch search.done
