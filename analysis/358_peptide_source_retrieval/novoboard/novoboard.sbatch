#!/bin/bash
#SBATCH --job-name=nb358-novoboard
#SBATCH --partition=hns,normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.log
# Experiment 358: download NovoBoard's human data (PRIDE PXD055277), human Swiss-Prot
# 2024_01 and cRAP; list what is in the zip; write calls.tsv, accepted.tsv, inputs.json.
# About 2.3 GB to download. The converter streams the zip and the tarball (under 1 GB RAM).
set -euo pipefail
: "${OUT:?set OUT}"
: "${BIN:?set BIN}"
PY=/usr/bin/python3
mkdir -p "$OUT/download" "$OUT/inputs"
cd "$OUT/download"

PRIDE=https://ftp.pride.ebi.ac.uk/pride/data/archive/2024/08/PXD055277
UNIPROT=https://ftp.uniprot.org/pub/databases/uniprot/previous_releases/release-2024_01/knowledgebase
declare -A URLS=(
  [eval_human.zip]=$PRIDE/eval_human.zip
  [uniprot_sprot-only2024_01.tar.gz]=$UNIPROT/uniprot_sprot-only2024_01.tar.gz
  [crap.fasta]=ftp://ftp.thegpm.org/fasta/cRAP/crap.fasta
)
# Sizes PRIDE and UniProt report; a download of another size stops the job.
declare -A BYTES=(
  [eval_human.zip]=710560551
  [uniprot_sprot-only2024_01.tar.gz]=1610292503
)

manifest=manifest.tsv
printf 'file\turl\tbytes\tlast_modified\tdate_downloaded\tsha256\n' > "$manifest"
for file in eval_human.zip uniprot_sprot-only2024_01.tar.gz crap.fasta; do
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
  if [[ -n "${BYTES[$file]:-}" && "$bytes" != "${BYTES[$file]}" ]]; then
    echo "$file is $bytes bytes, expected ${BYTES[$file]}" >&2
    exit 1
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$file" "$url" "$bytes" "$last_modified" \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(sha256sum "$file" | cut -d' ' -f1)" >> "$manifest"
done
cat "$manifest"

$PY "$BIN/convert_novoboard.py" inspect eval_human.zip > "$OUT/inputs/inspect.json"
$PY "$BIN/convert_novoboard.py" collection uniprot_sprot-only2024_01.tar.gz crap.fasta "$OUT/inputs"
$PY "$BIN/convert_novoboard.py" convert eval_human.zip "$OUT/inputs"
cp "$manifest" "$OUT/inputs/download_manifest.tsv"
(cd "$OUT/inputs" && sha256sum calls.tsv accepted.tsv proteins_human.fasta proteins_contaminants.fasta > checksums.sha256)
ls -la "$OUT/inputs"
