#!/bin/bash
#SBATCH --job-name=qs-pfam-breadth
#SBATCH --account=ayeletv
#SBATCH --partition=hns,normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/users/olgabot/leaderboard-query-strata/logs/%x-%j.log
#
# 5. Pfam family breadth: superkingdoms and phyla each family occurs in, over Pfam 38's
# sequences (UniProtKB reference proteomes 2025_03). See pfam_family_breadth.py.
# Ask: 4 CPUs, 64 GB, 3 h. History: filtering the same 5 GB Pfam-A.regions.tsv.gz took
# 2 minutes in scripts/nb293/filter_pfam_regions_swissprot.sbatch; the 24.7 GB pfamseq.gz
# has no history. The join holds ~90 M accessions in memory, hence 64 GB.
set -euo pipefail
source /scratch/users/olgabot/leaderboard-query-strata/repo/scripts/query_strata/paths.sh
REGIONS=$SCRATCH/data/pfam/regions/Pfam-A.regions.tsv.gz        # Pfam 38.2
PFAMSEQ=$SCRATCH/280-time-split/data/Pfam38.0/pfamseq.gz         # pfamseq 38 (same for 38.x)
OUT=$WD/out/pfam_breadth
TMP=$WD/tmp/pfam_breadth
mkdir -p "$OUT" "$TMP"
grep -n "Pfamseq 38\|reference proteomes 2025_03" \
  $SCRATCH/280-time-split/data/Pfam38.0/relnotes.txt $SCRATCH/280-time-split/data/Pfam38.2/relnotes.txt || true
ls -l "$REGIONS" "$PFAMSEQ"

tar -xzf "$WD/ref/new_taxdump.tar.gz" -C "$WD/ref" nodes.dmp
if [ ! -s "$TMP/regions.tsv" ]; then
  pigz -dc -p 4 "$REGIONS" | awk -F'\t' -v OFS='\t' 'NR>1{print $1, $5}' > "$TMP/regions.tsv.part"
  mv "$TMP/regions.tsv.part" "$TMP/regions.tsv"
fi
if [ ! -s "$TMP/mnemonics.tsv" ]; then
  # >A0A8J8BLT5.1 A0A8J8BLT5_9EURY Glutamyl-tRNA ...  ->  A0A8J8BLT5 <tab> 9EURY
  pigz -dc -p 4 "$PFAMSEQ" | grep '^>' \
    | awk -v OFS='\t' '{a=substr($1,2); sub(/\.[0-9]+$/,"",a); n=split($2,b,"_"); print a, b[n]}' \
    > "$TMP/mnemonics.tsv.part"
  mv "$TMP/mnemonics.tsv.part" "$TMP/mnemonics.tsv"
fi
wc -l "$TMP/regions.tsv" "$TMP/mnemonics.tsv"

apptainer exec --bind "$WD" "$PY_IMG" python3 \
  "$WD/repo/scripts/query_strata/pfam_family_breadth.py" \
  --regions "$TMP/regions.tsv" --mnemonics "$TMP/mnemonics.tsv" \
  --speclist "$WD/ref/speclist.txt" --nodes "$WD/ref/nodes.dmp" \
  --rankedlineage "$WD/ref/rankedlineage.dmp" --out "$OUT/pfam_family_breadth.tsv"
sha256sum "$REGIONS" "$PFAMSEQ" "$OUT/pfam_family_breadth.tsv" > "$OUT/pfam_breadth.sha256"
