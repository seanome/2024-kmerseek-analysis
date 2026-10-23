#!/bin/bash
#SBATCH --job-name=oyster-mvp-baseline
#SBATCH --partition=normal,hns
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=%x.%j.out
# Reference baseline at one depth (argument: 1m, 5m or full): kallisto quant
# on the same trimmed pairs as the discovery methods, gene aggregation,
# PyDESeq2 breed over wild. Used only for grading and for the comparison of
# what the reference route does not see.
set -euo pipefail
source "$HOME/oyster-mvp-code/config.sh"
depth=$1
reads="$ROOT/trimmed/$depth"
out="$ROOT/baseline/$depth"
mkdir -p "$out/quant"
for s in "${SAMPLES[@]}"; do
  [ -f "$reads/$s.validated.json" ] || { echo "trimmed $s $depth not validated" >&2; exit 1; }
done
printf 'sample\tcondition\n' > "$out/samples.tsv"
for s in "${SAMPLES[@]}"; do printf '%s\t%s\n' "$s" "${CONDITION[$s]}" >> "$out/samples.tsv"; done

for s in "${SAMPLES[@]}"; do
  [ -f "$out/quant/$s/run_info.json" ] && continue
  rm -rf "$out/quant/$s.part"
  timed "kallisto-quant-$depth-$s" "$SHAZAM_ENV/bin/kallisto" quant -t 8 \
    -i "$ROOT/references/oyster_v9.kallisto.idx" -o "$out/quant/$s.part" \
    "$reads/${s}_val_1.fq.gz" "$reads/${s}_val_2.fq.gz"
  mv "$out/quant/$s.part" "$out/quant/$s"
done

timed "baseline-aggregate-$depth" "$SHAZAM_ENV/bin/python" "$CODE/aggregate_baseline.py" \
  --feature-table "$ROOT/references/feature_table.txt.gz" --quant-dir "$out/quant" \
  --samples "$out/samples.tsv" --out "$out"
timed "baseline-de-$depth" "$SHAZAM_ENV/bin/python" "$CODE/sources/package_fit_de.py" \
  --counts "$out/gene_counts.tsv" --samples "$out/samples.tsv" --out "$out/de"
test -s "$out/de/de_results.tsv"
touch "$out/complete"
