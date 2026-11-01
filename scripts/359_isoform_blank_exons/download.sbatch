#!/bin/bash
# Notebook 359, data: every input on disk, checked against its published md5, before any
# compute. Writes data/manifest.tsv (file, URL, md5, bytes, date) for the notebook.
#
# Sources, each checked on 2026-10-20:
#   WTC11 long reads: ENCODE ENCSR507JOF, "LRGASP WTC11" PacBio Sequel II cDNA (Mortazavi
#     lab). The LRGASP copy on Synapse needs a login, so the ENCODE copy is used. Its
#     three TALON "transcriptome annotations" GTFs (one per replicate, GRCh38, built on
#     GENCODE V29) are joined by transcript ID in merge_gtfs.py.
#   GENCODE v39 primary-assembly genome and GTF, and the Swiss-Prot cross-references:
#     the release tappAS built its human GFF3 on (Homo_sapiens_Gencode_v39.zip).
#   tappAS human GFF3 for IsoAnnotLite: app.tappas.org (no md5 published; ours recorded).
#   InterProScan 5.78-109.0, the version proteome-interproscan/Makefile pins.
#   SQANTI3 v6.0.1 image, the newest SQANTI3 image on Docker Hub (GitHub's v6.1 tag has
#     no image).
#
# A file already present with the right md5 is not fetched again.
#SBATCH --job-name=359-download
#SBATCH --account=ayeletv
#SBATCH --partition=hns
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x.%j.log

set -euo pipefail
WORK=$(pwd)
D=$WORK/data
mkdir -p "$D" "$D/tappas"
CACHE=/scratch/users/olgabot/apptainer-cache
ENC=https://www.encodeproject.org/files
GC=https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_39
IPS_V=5.78-109.0
IPS=https://ftp.ebi.ac.uk/pub/software/unix/iprscan/5/$IPS_V

printf 'file\turl\tmd5\tbytes\tdate\n' > "$D/manifest.tsv"
fetch() {  # fetch <url> <file> <md5 or ->
    local url=$1 f=$D/$2 want=$3
    if [ ! -s "$f" ] || { [ "$want" != "-" ] && [ "$(md5sum < "$f" | cut -d' ' -f1)" != "$want" ]; }; then
        curl -sSfL --retry 5 -o "$f.part" "$url"
        mv "$f.part" "$f"
    fi
    local got
    got=$(md5sum < "$f" | cut -d' ' -f1)
    if [ "$want" != "-" ] && [ "$got" != "$want" ]; then
        echo "md5 mismatch for $f: got $got, published $want" >&2
        exit 1
    fi
    printf '%s\t%s\t%s\t%s\t%s\n' "$2" "$url" "$got" "$(stat -c %s "$f")" "$(date -u +%F)" >> "$D/manifest.tsv"
}

fetch "$ENC/ENCFF370SDY/@@download/ENCFF370SDY.gtf.gz" ENCFF370SDY.gtf.gz 45e7f7a314038b09829065defeb86263
fetch "$ENC/ENCFF798DWV/@@download/ENCFF798DWV.gtf.gz" ENCFF798DWV.gtf.gz 6f981bcacedcf9d2843ae3f1dceeebca
fetch "$ENC/ENCFF422CGK/@@download/ENCFF422CGK.gtf.gz" ENCFF422CGK.gtf.gz 2e50f7e5b95a6c505d8ddbcb00d17a9c
fetch "$GC/GRCh38.primary_assembly.genome.fa.gz" GRCh38.primary_assembly.genome.fa.gz 6e8f6b77ec0c31e6ccf0d0856619d209
fetch "$GC/gencode.v39.primary_assembly.annotation.gtf.gz" gencode.v39.primary_assembly.annotation.gtf.gz 35607b24b85b16b75ad9fab0a44a01a6
fetch "$GC/gencode.v39.metadata.SwissProt.gz" gencode.v39.metadata.SwissProt.gz 7203dd678d8a4efe386c222d615b9e44
fetch "http://app.tappas.org/resources/downloads/gffs/Homo_sapiens_Gencode_v39.zip" Homo_sapiens_Gencode_v39.zip -
fetch "$IPS/interproscan-$IPS_V-64-bit.tar.gz" "interproscan-$IPS_V-64-bit.tar.gz" 3bb9a0794e9d69a0418a5298cdb04445

# SQANTI3 reads an uncompressed genome and GTF.
[ -s "$D/GRCh38.primary_assembly.genome.fa" ] || gunzip -c "$D/GRCh38.primary_assembly.genome.fa.gz" > "$D/GRCh38.primary_assembly.genome.fa"
[ -s "$D/gencode.v39.primary_assembly.annotation.gtf" ] || gunzip -c "$D/gencode.v39.primary_assembly.annotation.gtf.gz" > "$D/gencode.v39.primary_assembly.annotation.gtf"
(cd "$D/tappas" && unzip -o -q ../Homo_sapiens_Gencode_v39.zip)
find "$D/tappas" -name '*.gff3' -exec sh -c 'echo "== $1"; head -5 "$1"; md5sum "$1"' _ {} \; | tee "$D/tappas/header.txt"

# InterProScan: unpack and press the HMMs once (the tarball ships them unpressed).
if [ ! -f "$D/interproscan-$IPS_V/.pressed" ]; then
    tar -xzf "$D/interproscan-$IPS_V-64-bit.tar.gz" -C "$D"
    ml python/3.12.1
    (cd "$D/interproscan-$IPS_V" && python3 setup.py -f interproscan.properties)
    touch "$D/interproscan-$IPS_V/.pressed"
fi

# SQANTI3 image
SQ=$CACHE/docker.io-anaconesalab-sqanti3-v6.0.1.img
[ -s "$SQ" ] || apptainer pull "$SQ" docker://anaconesalab/sqanti3:v6.0.1
apptainer exec "$SQ" bash -c 'ls /opt2/sqanti3/*/SQANTI3-*/sqanti3_qc.py'

# The three replicate GTFs joined by transcript ID, on the genome's chromosomes only.
apptainer exec --bind /scratch /scratch/users/olgabot/apptainer-cache/docker.io-olgabot-kmerseek-0.4.1-rc1.img \
    python3 code/scripts/359_isoform_blank_exons/merge_gtfs.py \
    --genome-fai-from "$D/GRCh38.primary_assembly.genome.fa" \
    --out "$D/wtc11_lrgasp_pacbio_union.gtf" \
    "$D/ENCFF370SDY.gtf.gz" "$D/ENCFF798DWV.gtf.gz" "$D/ENCFF422CGK.gtf.gz"
cat "$D/manifest.tsv"
