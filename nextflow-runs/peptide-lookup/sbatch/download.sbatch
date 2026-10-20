#!/bin/bash
#SBATCH --job-name=nb300-download
#SBATCH --partition=hns,normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=08:00:00
#SBATCH --output=%x-%j.log
# Download the notebook 300 databases into $OUT/db and record what was read and when.
# 32 GB UniRef90 and 11 GB UHGP-90 at a few MB/s is 2-4 h; curl resumes on a stall.
set -euo pipefail
: "${OUT:?set OUT}"
mkdir -p "$OUT/db"
cd "$OUT/db"

UNIPROT=https://ftp.uniprot.org/pub/databases/uniprot
MGNIFY=https://ftp.ebi.ac.uk/pub/databases/metagenomics/mgnify_genomes/human-gut/v2.0.2
declare -A URLS=(
  [uniprot_sprot.fasta.gz]=$UNIPROT/current_release/knowledgebase/complete/uniprot_sprot.fasta.gz
  [uniref90.fasta.gz]=$UNIPROT/uniref/uniref90/uniref90.fasta.gz
  [uhgp-90.tar.gz]=$MGNIFY/protein_catalogue/uhgp-90.tar.gz
  [human_UP000005640_9606.fasta.gz]=$UNIPROT/current_release/knowledgebase/reference_proteomes/Eukaryota/UP000005640/UP000005640_9606.fasta.gz
  [human_UP000005640_9606_additional.fasta.gz]=$UNIPROT/current_release/knowledgebase/reference_proteomes/Eukaryota/UP000005640/UP000005640_9606_additional.fasta.gz
  [crap.fasta]=ftp://ftp.thegpm.org/fasta/cRAP/crap.fasta
)

manifest=manifest.tsv
printf 'file\turl\tbytes\tlast_modified\tdate_downloaded\tsha256\n' > "$manifest"
for file in "${!URLS[@]}"; do
  url=${URLS[$file]}
  last_modified=$(curl -sIL "$url" | tr -d '\r' | awk -F': ' 'tolower($1)=="last-modified"{print $2}' | tail -1)
  for attempt in 1 2 3 4 5 6 7 8 9 10; do
    if curl -fsSL -C - --retry 5 --speed-limit 100000 --speed-time 60 -o "$file" "$url"; then
      break
    fi
    echo "attempt $attempt for $file stopped; resuming" >&2
    sleep 30
  done
  bytes=$(stat -c %s "$file")
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$file" "$url" "$bytes" "$last_modified" \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(sha256sum "$file" | cut -d' ' -f1)" >> "$manifest"
done

# UHGP-90 ships as a tar of the FASTA and its cluster tables; keep the FASTA gzipped.
tar -tzf uhgp-90.tar.gz > uhgp-90.contents.txt
faa=$(grep -E '\.faa(\.gz)?$' uhgp-90.contents.txt | head -1)
tar -xzf uhgp-90.tar.gz "$faa"
case "$faa" in
  *.gz) ln -sf "$faa" uhgp-90.faa.gz ;;
  *) gzip -c "$faa" > uhgp-90.faa.gz && command rm -f "$faa" ;;
esac
cat "$manifest"
