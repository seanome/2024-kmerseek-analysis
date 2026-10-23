#!/bin/bash
#SBATCH --job-name=299-setup
#SBATCH --partition=hns
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=299-setup-%j.out
#
# Notebook 299, setup: tools, the OM-RGC v2 gene catalog, its proteins, the 1 GB slice for
# step 1, and 256 catalog parts for step 3. Submitted by notebooks/299_submit_all.sh, which
# chains every job of the pilot.
#
# Catalog: BioStudies S-BSST297, "Ocean Microbial Reference Gene Catalog v2 (OM-RGC.v2)",
# released 2019-11-12, file OM-RGC_v2.tsv.gz, 10_320_406_416 bytes (listing read
# 2026-10-18). No md5 is published; gzip's full check and the byte count are used instead.
#
# Sizing: the download is 10.3 GB; 32 connections took about 30 min for 29.6 GB in job
# 48522928, so about 10 min; gunzip with 8 pigz threads and a six-figure-GB catalog are minutes to an hour;
# seqkit translate with 8 threads on about 40 GB of DNA is under an hour. 12 h is a wide
# margin; 32 G covers seqkit's buffers (it streams). Read the real use afterwards with
# `sacct -j <jobid> -o JobID,MaxRSS,Elapsed`.
set -euo pipefail

REPO=${SLURM_SUBMIT_DIR:?submit from the repo root}
WORK=${OMRGC_WORK:-$SCRATCH/299-omrgc-pilot}
# The catalog comes from OM-RGC_v2.tsv.gz (the gene table; column 14 is each gene's DNA),
# not OM-RGC_v2_genes.fna.gz: two parallel downloads of the .fna.gz (jobs 48522928 and
# 48546051) differed at byte 3_621_226_795 and the first failed gzip's check, while the
# .tsv.gz passed gzip's full check. Its first 40_000 genes translate in frame 1 with no
# internal stop.
URL=https://ftp.ebi.ac.uk/biostudies/fire/S-BSST/S-BSSTxxx297/S-BSST297/Files/OM-RGC_v2.tsv.gz
EXPECTED_BYTES=10320406416
SLICE_BYTES=1000000000
STEP1_TIME_LIMIT_S=40000
N_PARTS=256
mkdir -p "$WORK"/{bin,catalog,slice,parts,sample,step1,diamond,search,catalog_sketch}
echo "host $(hostname); $(lscpu | grep 'Model name')"

# 1. Tools: micromamba 2.9.0-0 (GitHub release, 2026-08-07) and the pinned env.
SOFTWARE=${OMRGC_SOFTWARE_DIR:-${GROUP_HOME:?}/software/omrgc299}
ENV_PREFIX=${OMRGC_ENV_PREFIX:-${GROUP_HOME:?}/conda/envs/omrgc299}
mkdir -p "$SOFTWARE"
if [ ! -x "$SOFTWARE/micromamba" ]; then
    curl -fsSL -o "$SOFTWARE/micromamba" \
        https://github.com/mamba-org/micromamba-releases/releases/download/2.9.0-0/micromamba-linux-64
    chmod +x "$SOFTWARE/micromamba"
fi
export MAMBA_ROOT_PREFIX=$SOFTWARE/mamba
if [ ! -x "$ENV_PREFIX/bin/sourmash" ]; then
    "$SOFTWARE/micromamba" create -y -p "$ENV_PREFIX" -f "$REPO/notebooks/299_environment.yml"
fi
E=${OMRGC_ENV_PREFIX:-${GROUP_HOME:?}/conda/envs/omrgc299}/bin
# A solve proves nothing about running: run every tool once.
"$E/diamond" version
"$E/seqkit" version
"$E/sourmash" --version
"$E/sourmash" scripts manysearch --help > /dev/null
"$E/python" -c "import polars, matplotlib, sourmash; print('polars', polars.__version__)"

# 2. Download as N_RANGES byte ranges in parallel (EBI serves each connection at about
#    1.9 MB/s, measured 2026-10-19). A piece is appended only when the server answers 206
#    (exactly the bytes asked for); a 200 would be the whole file from byte 0, which is the
#    likely way the .fna.gz download was silently damaged. Each range resumes on a stall
#    (under 100 kB/s for 60 s). The joined file must pass gzip's full check, which covers
#    every byte; otherwise it is fetched again, up to 3 times.
GZ=$WORK/catalog/OM-RGC_v2.tsv.gz
N_RANGES=32
fetch_range() {  # fetch_range INDEX FIRST_BYTE LAST_BYTE
    local part=$WORK/catalog/range.$1 want=$(( $3 - $2 + 1 )) have code
    : > "$part"
    for attempt in $(seq 1 50); do
        have=$(stat -c %s "$part")
        [ "$have" -eq "$want" ] && return 0
        code=$(curl -sL --speed-limit 100000 --speed-time 60 -r "$(( $2 + have ))-$3" \
            -o "$part.try" -w '%{http_code}' "$URL" || true)
        if [ "$code" = 206 ]; then cat "$part.try" >> "$part"; fi
        rm -f "$part.try"
    done
    echo "range $1: $(stat -c %s "$part") of $want bytes after 50 attempts"; return 1
}
download() {
    rm -f "$GZ" "$WORK"/catalog/range.*
    local step=$(( (EXPECTED_BYTES + N_RANGES - 1) / N_RANGES )) i first last pids=()
    for i in $(seq 0 $(( N_RANGES - 1 ))); do
        first=$(( i * step )); last=$(( first + step - 1 ))
        (( last >= EXPECTED_BYTES )) && last=$(( EXPECTED_BYTES - 1 ))
        fetch_range "$i" "$first" "$last" &
        pids+=($!)
    done
    for pid in "${pids[@]}"; do wait "$pid"; done
    for i in $(seq 0 $(( N_RANGES - 1 ))); do cat "$WORK/catalog/range.$i"; done > "$GZ"
    rm -f "$WORK"/catalog/range.*
}
gz_ok() { [ "$(stat -c %s "$GZ" 2>/dev/null || echo 0)" = "$EXPECTED_BYTES" ] && "$E/pigz" -t "$GZ"; }
# Check job 48546051 already downloaded and verified a copy; use it if it is there.
if [ ! -e "$GZ" ] && [ -e "$WORK/check/OM-RGC_v2.tsv.gz" ]; then mv "$WORK/check/OM-RGC_v2.tsv.gz" "$GZ"; fi
for round in 1 2 3; do
    gz_ok && break
    echo "download round $round"
    download
done
gz_ok || { echo "$GZ failed gzip's check after 3 downloads"; exit 1; }

# 3. FASTA from the table: name = gene (column 1) and OM-RGC_ID (column 2), sequence =
#    column 14. The first line is the header.
FNA=$WORK/catalog/OM-RGC_v2_genes_from_tsv.fna
if [ ! -s "$FNA" ]; then
    "$E/pigz" -dc -p 8 "$GZ" | awk -F'\t' 'NR > 1 { print ">" $1 " " $2; print $14 }' > "$FNA.tmp"
    mv "$FNA.tmp" "$FNA"
fi
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
