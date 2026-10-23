#!/bin/bash
#SBATCH --job-name=nb354-download
#SBATCH --partition=hns,normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.log
# Download the notebook 354 inputs into $OUT/raw and record what was read and when.
#
#   PRIDE PXD005280 (Blank-Landeshammer et al. 2017, HeLa / C2C12 / yeast on a Q Exactive HF):
#     PEAKS.zip      PEAKS de novo calls for every run, with per-residue local confidence
#     HeLa_1_qExHF01_02580.msf   Proteome Discoverer database search of the same HeLa run
#   UniProt human reference proteome UP000005640 (canonical, one entry per protein), current release
#   cRAP contaminant list (The GPM)
# About 330 MB in total. curl resumes on a stall.
set -euo pipefail
: "${OUT:?set OUT}"
mkdir -p "$OUT/raw"
cd "$OUT/raw"

PRIDE=https://ftp.pride.ebi.ac.uk/pride/data/archive/2019/09/PXD005280
UNIPROT=https://ftp.uniprot.org/pub/databases/uniprot/current_release
declare -A URLS=(
  [PXD005280_PEAKS.zip]=$PRIDE/PEAKS.zip
  [PXD005280_HeLa_1_qExHF01_02580.msf]=$PRIDE/HeLa_1_qExHF01_02580.msf
  [UP000005640_9606.fasta.gz]=$UNIPROT/knowledgebase/reference_proteomes/Eukaryota/UP000005640/UP000005640_9606.fasta.gz
  [uniprot_relnotes.txt]=$UNIPROT/relnotes.txt
  [crap.fasta]=ftp://ftp.thegpm.org/fasta/cRAP/crap.fasta
)

manifest=manifest.tsv
printf 'file\turl\tbytes\tlast_modified\tdate_downloaded\tsha256\n' > "$manifest"
for file in $(printf '%s\n' "${!URLS[@]}" | sort); do
  url=${URLS[$file]}
  last_modified=$(curl -sIL "$url" | tr -d '\r' | awk -F': ' 'tolower($1)=="last-modified"{print $2}' | tail -1 || true)
  for attempt in 1 2 3 4 5 6; do
    if curl -fsSL -C - --retry 5 --speed-limit 10000 --speed-time 60 -o "$file" "$url"; then
      break
    fi
    echo "attempt $attempt for $file stopped; resuming" >&2
    sleep 30
  done
  test -s "$file"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$file" "$url" "$(stat -c %s "$file")" "$last_modified" \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(sha256sum "$file" | cut -d' ' -f1)" >> "$manifest"
done
unzip -o -q PXD005280_PEAKS.zip -d PEAKS
cat "$manifest"
head -5 uniprot_relnotes.txt
