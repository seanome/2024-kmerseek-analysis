#!/bin/bash
#SBATCH --job-name=oyster-mvp-pilot-prefix
#SBATCH --partition=normal,hns
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=%x.%j.out
# The 1M pilot starts before the full ENA files finish downloading (about
# 2 MB/s from Sherlock). Input: the first 1,000,000 records of each ENA file,
# already fetched by the Codex run in $SCRATCH/oyster-shazam-20261023/raw
# (ENA byte prefixes, recompressed). Each copy is accepted only if its
# uncompressed sha256 matches the value recorded there and it holds exactly
# 4,000,000 lines. check_prefix.sbatch later confirms each one equals the
# first 4,000,000 lines of the md5-verified full file.
set -euo pipefail
source "$HOME/oyster-mvp-code/config.sh"
src="$SCRATCH/oyster-shazam-20261023/raw"
dst="$ROOT/prefix/1m"
mkdir -p "$dst"
for s in "${SAMPLES[@]}"; do
  for m in 1 2; do
    f="${s}_$m.fastq.gz"
    want=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["uncompressed_sha256"])' "$src/$s/${s}_$m.fastq.json")
    cp "$src/$s/$f" "$dst/$f.part"
    got=$(zcat "$dst/$f.part" | sha256sum | cut -d' ' -f1)
    lines=$(zcat "$dst/$f.part" | wc -l)
    [ "$got" = "$want" ] && [ "$lines" -eq 4000000 ] || { echo "MISMATCH $f $got $lines" >&2; exit 1; }
    mv "$dst/$f.part" "$dst/$f"
    printf '%s\t%s\t%s\n' "$f" "$got" "$lines" >> "$dst/sha256.tsv"
  done
done
touch "$dst/complete"
