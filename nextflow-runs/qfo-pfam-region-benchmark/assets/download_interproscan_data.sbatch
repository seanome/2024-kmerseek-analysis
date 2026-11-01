#!/bin/bash
# InterProScan 5.78-109.0's full distribution (software + member database data), into
# $SCRATCH/data/interproscan/interproscan-5.78-109.0, then its one-time index step
# (setup.py: hmmpress and friends). Submitted by `make interproscan-data`.
#
# interproscan-5.78-109.0-64-bit.tar.gz is 7_037_746_512 bytes (EBI's Content-Length,
# 2026-10-16); the md5 file beside it is checked. The unpacked size is printed by du.
#
# Run in the same image the pipeline uses, so setup.py indexes with the tools that will
# read the indexes.
#SBATCH --job-name=interproscan-data
#SBATCH --account=ayeletv
#SBATCH --partition=normal
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=interproscan-data-%j.out
set -euo pipefail

VER=5.78-109.0
BASE=https://ftp.ebi.ac.uk/pub/software/unix/iprscan/5/$VER
IMAGE=docker://docker.io/interpro/interproscan@sha256:dc58b7c147fbbf00c2dd4f5ced42121fc1e8841fcbc7cc2c484380248ff76d11
OUT=$SCRATCH/data/interproscan
TAR=interproscan-$VER-64-bit.tar.gz
mkdir -p "$OUT"
cd "$OUT"

for attempt in 1 2 3 4 5 6 7 8; do
    if curl -fL -C - --retry 3 --speed-limit 10000 --speed-time 120 -o "$TAR" "$BASE/$TAR"; then
        break
    fi
    echo "attempt $attempt stopped; retrying"
    sleep 30
done
curl -fsSL -o "$TAR.md5" "$BASE/$TAR.md5"
md5sum -c "$TAR.md5"

[ -d "interproscan-$VER" ] || tar -pxzf "$TAR"
cd "interproscan-$VER"
apptainer exec --bind "$SCRATCH:$SCRATCH" "$IMAGE" \
    python3 setup.py -f interproscan.properties
du -sh "$OUT/interproscan-$VER" "$OUT/interproscan-$VER/data"
