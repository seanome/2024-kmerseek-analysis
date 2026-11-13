#!/usr/bin/env bash
# P66 (Borreliella burgdorferi, mature chain 22-618) against human GENCODE v49,
# across alphabets and k-mer sizes, plus direct pair tests against the receptor
# families P66 is documented to bind.
#
#   bash p66_alphabet_ladder.sh <kmerseek-binary> <outdir>
#
# The mismatch penalty is not guessed. For an alphabet whose background match
# probability is u (the chance two unrelated residues land in the same class), a
# match stops being evidence at u = C/(1+C), so C = u/(1-u) is the boundary.
# hp_lehninger2 is anchored at C = 1.63 from notebook 230, which is 1.63x its own
# u/(1-u); every other alphabet keeps that same ratio, and X-drop stays at 4C.
set -euo pipefail

KS="${1:?path to the kmerseek binary}"
OUT="${2:-out}"
GEN="$HOME/data/gencode/human/v49/gencode.v49.pc_translations.canonical.fa"
P66="$OUT/P66.fa"
mkdir -p "$OUT"

# alphabet:ksize:mismatch-penalty:xdrop
ARMS=(
  "hp_lehninger2:24:1.63:6.52"
  "protein20:13:0.14:0.56"
  "funcgroups8:15:0.365:1.460"
  "funcgroups8:8:0.365:1.460"
  "polarity4:8:0.794:3.177"
)

for arm in "${ARMS[@]}"; do
  IFS=: read -r alphabet ksize penalty xdrop <<< "$arm"
  idx="$OUT/gencode.$alphabet.k$ksize.rocksdb"
  echo "=== $alphabet k=$ksize  C=$penalty X=$xdrop ==="

  # The Karlin-Altschul fit is refused for some alphabet and k combinations:
  # "too few score bins" is a score-spread failure, not a query shortage, so
  # raising --ka-queries does not help. Fall back to a search without extension,
  # which needs no fit and still answers whether anything matches at all.
  if [[ ! -d "$idx" ]]; then
    "$KS" index --input "$GEN" --output "$idx" \
      --alphabet "$alphabet" --ksize "$ksize" --scaled 1 --remove-low-complexity \
      --extend-mismatch-penalty "$penalty" --extend-xdrop "$xdrop" \
      --ka-queries 200 --ka-seed 1 --ka-reference-shuffles 20 \
      --ka-survival-out "$OUT/ka_survival.$alphabet.k$ksize.csv"
  fi

  if "$KS" search --query "$P66" --target "$idx" \
       --alphabet "$alphabet" --ksize "$ksize" \
       --extend-mismatch-penalty "$penalty" --extend-xdrop "$xdrop" \
       --max-query-pvalue 0.05 --min-region-score 1.3 \
       --output "$OUT/P66.$alphabet.k$ksize.csv"; then
    echo "  searched with extension"
  else
    echo "  no Karlin-Altschul fit for this combination, searching without extension"
    "$KS" search --query "$P66" --target "$idx" \
      --alphabet "$alphabet" --ksize "$ksize" \
      --max-query-pvalue 0.05 --min-region-score 1.3 \
      --output "$OUT/P66.$alphabet.k$ksize.csv"
  fi
done

# Direct pair tests. kmerseek grows a region out of an exact k-mer seed, so a
# pair with no shared k-mer can have no region however the extension is set:
# these zeros are not a threshold effect. The P66-against-itself control shows
# the test works.
echo "=== pair tests against the receptors P66 is documented to bind ==="
for gene in ITGB3 ITGAV ITGB1 ITGA3 CD47; do
  for spec in "hp_lehninger2:24" "protein20:13" "protein20:10" "funcgroups8:8" "polarity4:8"; do
    IFS=: read -r alphabet ksize <<< "$spec"
    "$KS" pair --query "$P66" --target "$OUT/receptors.fa" \
      --query-name P66 --target-name "$gene" \
      --alphabet "$alphabet" --ksize "$ksize" \
      --output "$OUT/pair.$gene.$alphabet.$ksize.json" >/dev/null
  done
done
"$KS" pair --query "$P66" --target "$P66" --query-name P66 --target-name P66 \
  --alphabet hp_lehninger2 --ksize 24 --output "$OUT/pair.self.json" >/dev/null
echo "control, P66 against itself: $(python3 -c "
import json; print(len(json.load(open('$OUT/pair.self.json'))['shared_kmers']), 'shared k-mers')")"
