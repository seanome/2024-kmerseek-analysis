#!/bin/bash
#SBATCH --job-name=oyster-mvp-check-prefix
#SBATCH --partition=normal,hns
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G
#SBATCH --time=01:00:00
#SBATCH --output=%x.%j.out
# After the full files pass md5: the 1M pilot input must equal the first
# 4,000,000 lines of each verified full file. Writes prefix/1m/full_match.tsv.
set -euo pipefail
source "$HOME/oyster-mvp-code/config.sh"
d="$ROOT/prefix/1m"
: > "$d/full_match.tsv.part"
bad=0
for s in "${SAMPLES[@]}"; do
  for m in 1 2; do
    f="${s}_$m.fastq.gz"
    set +o pipefail
    full=$(zcat "$ROOT/raw/$f" | head -n 4000000 | sha256sum | cut -d' ' -f1)
    set -o pipefail
    pilot=$(awk -F'\t' -v f="$f" '$1==f{print $2}' "$d/sha256.tsv")
    same=$([ "$full" = "$pilot" ] && echo yes || echo NO)
    [ "$same" = yes ] || bad=1
    printf '%s\t%s\t%s\t%s\n' "$f" "$pilot" "$full" "$same" >> "$d/full_match.tsv.part"
  done
done
mv "$d/full_match.tsv.part" "$d/full_match.tsv"
cat "$d/full_match.tsv"
exit $bad
