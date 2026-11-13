#!/bin/bash
#SBATCH --job-name=nb361-sra-search
#SBATCH --account=ayeletv
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x.%j.out
# Sketch the query proteins (one signature per protein) and search them against every run
# signature, per cutoff and setting. Threshold 0 with -A: every query-run pair is reported,
# including zero overlap. QUERIES names the FASTA (queries/queries.faa or pilot_queries.faa);
# TAG names the output folder.
set -euo pipefail
: "${PEC_ROOT:?}" "${CUTOFFS:?}" "${QUERIES:?}" "${TAG:?}"
export PATH=$PEC_ROOT/env/bin:$PATH
W=$PEC_ROOT/sra_search
cd "$W"
mkdir -p "search_$TAG"
cd "search_$TAG"
printf 'name,genome_filename,protein_filename\nqueries,,%s\n' "$W/$QUERIES" > q.csv
sourmash scripts manysketch q.csv --singleton -c "$SLURM_CPUS_PER_TASK" \
  -p hp,k=27,scaled=10 -p hp,k=30,scaled=10 -p dayhoff,k=12,scaled=10 -p dayhoff,k=13,scaled=10 \
  -p protein,k=6,scaled=10 -p protein,k=7,scaled=10 -o queries.sig.zip > queries.sketch.log 2>&1
# hashes per signature, so an empty query or run sketch reads as "unsearchable", not "absent"
sourmash sig describe --csv queries.describe.csv queries.sig.zip > /dev/null 2>&1
for c in ${CUTOFFS//,/ }; do
  ls "$W"/runs/*/*.ge$c.sig.zip > against.ge$c.txt
  sourmash sig describe --csv runs.ge$c.describe.csv $(cat against.ge$c.txt) > /dev/null 2>&1
  for spec in hp:27 hp:30 dayhoff:12 dayhoff:13 protein:6 protein:7; do
    m=${spec%:*}; k=${spec#*:}
    out=ms.ge$c.$m.k$k.csv
    [ -s "$out" ] && continue
    /usr/bin/time -v -o "$out.time" sourmash scripts manysearch queries.sig.zip against.ge$c.txt \
      -m "$m" -k "$k" -s 10 -t 0 -A --ignore-abundance -N -c "$SLURM_CPUS_PER_TASK" -o "$out.part" \
      > "$out.log" 2>&1
    mv "$out.part" "$out"
  done
done
touch search.done
