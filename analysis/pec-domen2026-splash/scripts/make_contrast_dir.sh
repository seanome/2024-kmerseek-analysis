#!/bin/bash
# Lay out $PEC_ROOT/contrasts/<contrast>/ from the frozen design (design/contrast_<name>.*):
# input.txt (sample, interleaved trimmed FASTQ), cjs.csv, samples.txt, design.tsv.
# Stops if any sample of the contrast is not trimmed and verified yet.
set -euo pipefail
: "${PEC_ROOT:?}" "${PEC_CODE:?}"
C=$1
D=$PEC_ROOT/contrasts/$C
mkdir -p "$D"
cp "$PEC_CODE/design/contrast_$C.design.tsv" "$D/design.tsv"
cp "$PEC_CODE/design/contrast_$C.cjs" "$D/cjs.csv"
tail -n +2 "$D/design.tsv" | cut -f1 > "$D/samples.txt"
: > "$D/input.txt"
while read -r s; do
  f=$PEC_ROOT/trimmed/$s.il.fastq.gz
  [ -e "$PEC_ROOT/trimmed/$s.done" ] && [ -s "$f" ] || { echo "$s not trimmed yet"; exit 2; }
  echo "$s $f" >> "$D/input.txt"
done < "$D/samples.txt"
[ "$(wc -l < "$D/cjs.csv")" = "$(wc -l < "$D/samples.txt")" ] || { echo "cjs and samples differ"; exit 3; }
echo "$C: $(wc -l < "$D/samples.txt") samples, $(awk '{s+=$1} END{print s}' $(sed "s#^#$PEC_ROOT/trimmed/#; s#\$#.done#" "$D/samples.txt")) trimmed reads"
