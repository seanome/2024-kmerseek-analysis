#!/bin/bash
#SBATCH --job-name=299-setup
#SBATCH --account=ayeletv
#SBATCH --partition=hns
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=299-setup-%j.out
#
# Notebook 299, setup: tools, the OM-RGC v2 gene catalog, its proteins, the 1 GB slice for
# step 1, and 256 catalog parts for step 3. Submit from the repo root of a checkout of
# olgabot/nb299-omrgc-sketch-pilot:
#
#   cd $SCRATCH/2024-kmerseek-analysis-299 && git pull && sbatch notebooks/299_setup.sbatch
#
# Catalog: BioStudies S-BSST297, "Ocean Microbial Reference Gene Catalog v2 (OM-RGC.v2)",
# released 2019-11-12, file OM-RGC_v2_genes.fna.gz, 29_607_178_264 bytes (listing and HTTP
# Content-Length read 2026-10-18). No md5 is published; the byte count is checked instead.
#
# Sizing (arithmetic, nothing measured yet): the download is 29.6 GB, at 50-100 MB/s about
# 5-10 min; gunzip with 8 pigz threads and a six-figure-GB catalog are minutes to an hour;
# seqkit translate with 8 threads on about 40 GB of DNA is under an hour. 12 h is a wide
# margin; 32 G covers seqkit's buffers (it streams). Read the real use afterwards with
# `sacct -j <jobid> -o JobID,MaxRSS,Elapsed`.
set -euo pipefail

REPO=${SLURM_SUBMIT_DIR:?submit from the repo root}
WORK=${OMRGC_WORK:-$SCRATCH/299-omrgc-pilot}
URL=https://ftp.ebi.ac.uk/biostudies/fire/S-BSST/S-BSSTxxx297/S-BSST297/Files/OM-RGC_v2_genes.fna.gz
EXPECTED_BYTES=29607178264
SLICE_BYTES=1000000000
STEP1_TIME_LIMIT_S=40000
N_PARTS=256
mkdir -p "$WORK"/{bin,catalog,slice,parts,sample,step1,diamond,search,catalog_sketch}
echo "host $(hostname); $(lscpu | grep 'Model name')"

# 1. Tools: micromamba 2.9.0-0 (GitHub release, 2026-08-07) and the pinned env.
if [ ! -x "$WORK/bin/micromamba" ]; then
    curl -fsSL -o "$WORK/bin/micromamba" \
        https://github.com/mamba-org/micromamba-releases/releases/download/2.9.0-0/micromamba-linux-64
    chmod +x "$WORK/bin/micromamba"
fi
export MAMBA_ROOT_PREFIX=$WORK/mamba
if [ ! -x "$WORK/env/bin/sourmash" ]; then
    "$WORK/bin/micromamba" create -y -p "$WORK/env" -f "$REPO/notebooks/299_environment.yml"
fi
E=$WORK/env/bin
# A solve proves nothing about running: run every tool once.
"$E/diamond" version
"$E/seqkit" version
"$E/sourmash" --version
"$E/sourmash" scripts manysearch --help > /dev/null
"$E/python" -c "import polars, matplotlib, sourmash; print('polars', polars.__version__)"

# 2. Download, resumable, then check the byte count.
GZ=$WORK/catalog/OM-RGC_v2_genes.fna.gz
if [ "$(stat -c %s "$GZ" 2>/dev/null || echo 0)" != "$EXPECTED_BYTES" ]; then
    curl -fL -C - --retry 5 -o "$GZ" "$URL"
fi
test "$(stat -c %s "$GZ")" = "$EXPECTED_BYTES" || { echo "size mismatch on $GZ"; exit 1; }
"$E/pigz" -t "$GZ"

# 3. Decompress and count.
FNA=$WORK/catalog/OM-RGC_v2_genes.fna
if [ ! -s "$FNA" ]; then "$E/pigz" -dc -p 8 "$GZ" > "$FNA.tmp" && mv "$FNA.tmp" "$FNA"; fi
GENES=$(grep -c '^>' "$FNA")
echo "genes in catalog: $GENES"

# 4. Proteins for DIAMOND: genetic code 11, frame 1, trailing stop trimmed, one line each.
#    OM-RGC genes are gene calls, so frame 1 should be in frame; internal stops say otherwise.
FAA=$WORK/catalog/OM-RGC_v2_proteins.faa
if [ ! -s "$FAA" ]; then
    "$E/seqkit" translate -j 8 -T 11 -f 1 --trim --allow-unknown-codon -w 0 "$FNA" > "$FAA.tmp" && mv "$FAA.tmp" "$FAA"
fi
WITH_STOP=$(grep -v '^>' "$FAA" | grep -c '\*' || true)
echo "proteins with an internal stop: $WITH_STOP of $GENES"

# 5. Step 1 slice: the catalog's first records up to 1 GB, cut at a record boundary.
SLICE=$WORK/slice/OM-RGC_v2_genes_first_1GB.fna
awk -v lim="$SLICE_BYTES" '/^>/{ if (n >= lim) exit } { print; n += length($0) + 1 }' "$FNA" > "$SLICE"
SLICE_GENES=$(grep -c '^>' "$SLICE")

# 6. Catalog parts for step 3 (each step 3 task takes the parts whose index mod N is its own).
if [ -z "$(ls "$WORK/parts" 2>/dev/null)" ]; then
    "$E/seqkit" split2 -j 8 -p "$N_PARTS" -O "$WORK/parts" "$FNA"
fi

# 7. A 1% random sample of catalog genes (seed 299), in 32 parts. Step 3 counts hits on it
#    when a setting's full-catalog output would be too large to stream (see 299_data/rules.md).
SAMPLE=$WORK/sample/OM-RGC_v2_genes_random_1pct.fna
mkdir -p "$WORK/sample/parts"
if [ ! -s "$SAMPLE" ]; then
    grep '^>' "$FNA" | cut -c2- | cut -d' ' -f1 > "$WORK/catalog/gene_names.txt"
    "$E/python" - "$WORK/catalog/gene_names.txt" "$WORK/sample/sample_names.txt" <<'PY'
import random, sys
names = open(sys.argv[1]).read().split()
keep = random.Random(299).sample(range(len(names)), len(names) // 100)
open(sys.argv[2], "w").write("\n".join(names[i] for i in sorted(keep)) + "\n")
PY
    "$E/seqkit" grep -j 8 -f "$WORK/sample/sample_names.txt" "$FNA" > "$SAMPLE"
    "$E/seqkit" split2 -j 8 -p 32 -O "$WORK/sample/parts" "$SAMPLE"
fi
SAMPLE_GENES=$(grep -c '^>' "$SAMPLE")

cat > "$WORK/catalog_stats.json" <<EOF
{
 "source_url": "$URL",
 "source_bytes_gz": $EXPECTED_BYTES,
 "downloaded": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
 "catalog_fna_bytes": $(stat -c %s "$FNA"),
 "catalog_genes": $GENES,
 "proteins_with_internal_stop": $WITH_STOP,
 "slice_bytes": $(stat -c %s "$SLICE"),
 "slice_genes": $SLICE_GENES,
 "step1_time_limit_s": $STEP1_TIME_LIMIT_S,
 "n_parts": $N_PARTS,
 "sample_genes": $SAMPLE_GENES
}
EOF
cat "$WORK/catalog_stats.json"
mkdir -p "$REPO/notebooks/299_data/sherlock"
cp "$WORK/catalog_stats.json" "$REPO/notebooks/299_data/sherlock/"
