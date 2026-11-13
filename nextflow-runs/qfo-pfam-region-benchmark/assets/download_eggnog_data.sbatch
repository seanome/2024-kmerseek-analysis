#!/bin/bash
# eggNOG-mapper's database (emapperdb 5.0.2, the version eggNOG-mapper 2.1.x reads), into
# $SCRATCH/data/eggnog/emapperdb-5.0.2. Submitted by `make eggnog-data`.
#
# Sizes from the server's Content-Length on 2026-10-16:
#   eggnog.db.gz             6_776_977_123 bytes (annotations, SQLite)
#   eggnog_proteins.dmnd.gz  5_208_806_170 bytes (the DIAMOND database emapper searches)
#   eggnog.taxa.tar.gz          72_797_584 bytes
# The unpacked size is printed at the end (du -sh), so the PR can state it measured.
#
# Resumable: curl -C - continues a partial file, and a finished file is skipped. A stall
# (under 10 kB/s for 2 minutes) aborts that attempt and the loop retries.
#SBATCH --job-name=eggnog-data
#SBATCH --account=ayeletv
#SBATCH --partition=normal
#SBATCH --time=04:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2
#SBATCH --output=eggnog-data-%j.out
set -euo pipefail

BASE=http://eggnog5.embl.de/download/emapperdb-5.0.2
OUT=$SCRATCH/data/eggnog/emapperdb-5.0.2
mkdir -p "$OUT"
cd "$OUT"

fetch() {
    local f=$1
    local done_marker=".$f.done"
    if [ -f "$done_marker" ]; then echo "have $f"; return; fi
    for attempt in 1 2 3 4 5 6 7 8; do
        if curl -fL -C - --retry 3 --speed-limit 10000 --speed-time 120 -o "$f" "$BASE/$f"; then
            touch "$done_marker"
            return
        fi
        echo "attempt $attempt for $f stopped; retrying"
        sleep 30
    done
    echo "could not download $f" >&2
    exit 1
}

for f in eggnog.taxa.tar.gz eggnog_proteins.dmnd.gz eggnog.db.gz; do fetch "$f"; done
ls -l eggnog.taxa.tar.gz eggnog_proteins.dmnd.gz eggnog.db.gz

[ -f eggnog.taxa.db ] || tar -xzf eggnog.taxa.tar.gz
# gzip on Sherlock is 1.5, which has no -k.
for f in eggnog_proteins.dmnd eggnog.db; do
    [ -f "$f" ] || { gunzip -c "$f.gz" > "$f.part" && mv "$f.part" "$f"; }
done
ls -l
du -sh "$OUT"
