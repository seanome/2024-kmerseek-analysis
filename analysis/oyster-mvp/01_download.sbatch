#!/bin/bash
#SBATCH --job-name=oyster-mvp-download
#SBATCH --partition=normal,hns
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=06:00:00
#SBATCH --array=0-6
#SBATCH --output=%x.%A.%a.out
# Tasks 0-5: both full FASTQ files of one library from ENA, md5 checked.
# Task 6: oyster_v9 rna/protein/feature table (sha256) and Swiss-Prot (md5).
# A file is written as .part and renamed only after its checksum matches,
# so later jobs never read a file that is still being written.
set -euo pipefail
source "$HOME/oyster-mvp-code/config.sh"
mkdir -p "$ROOT/raw" "$ROOT/references"

fetch() {  # url dest algorithm expected
  local url=$1 dest=$2 algo=$3 want=$4 got
  if [ -f "$dest" ]; then echo "present $dest"; return 0; fi
  # ENA transfers can stall silently; abort under 1 KB/s for 60 s and resume.
  local attempt
  for attempt in $(seq 1 30); do
    curl -fsSL --retry 3 --speed-limit 1024 --speed-time 60 -C - -o "$dest.part" "$url" && break
    echo "attempt $attempt for $url stopped; resuming" >&2
    sleep 10
  done
  got=$("${algo}sum" "$dest.part" | cut -d' ' -f1)
  if [ "$got" != "$want" ]; then
    echo "CHECKSUM MISMATCH $url: $got != $want" >&2
    mv "$dest.part" "$dest.bad"
    return 1
  fi
  mv "$dest.part" "$dest"
  echo "verified $algo $got $dest"
}

i=$SLURM_ARRAY_TASK_ID
if [ "$i" -lt 6 ]; then
  sample=${SAMPLES[$i]}
  row=$(awk -F'\t' -v s="$sample" '$3==s' "$CODE/sources/oyster_runs.tsv")
  run=$(cut -f1 <<<"$row")
  IFS=';' read -r url1 url2 <<<"$(cut -f7 <<<"$row")"
  IFS=';' read -r md51 md52 <<<"$(cut -f9 <<<"$row")"
  timed_fn "download-$sample-1" fetch "https://$url1" "$ROOT/raw/${sample}_1.fastq.gz" md5 "$md51" &
  pid1=$!
  timed_fn "download-$sample-2" fetch "https://$url2" "$ROOT/raw/${sample}_2.fastq.gz" md5 "$md52" &
  pid2=$!
  wait $pid1; wait $pid2
  printf '%s\t%s\t%s\t%s\t%s\n' "$sample" "$run" "$url1" "$md51" "$md52" > "$ROOT/raw/$sample.verified.tsv"
else
  cd "$ROOT/references"
  for f in rna.fna.gz protein.faa.gz feature_table.txt.gz; do
    url=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))[sys.argv[2]]['url'])" "$CODE/sources/REFERENCE.json" "$f")
    sha=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))[sys.argv[2]]['sha256'])" "$CODE/sources/REFERENCE.json" "$f")
    timed_fn "download-ref-$f" fetch "$url" "$ROOT/references/$f" sha256 "$sha"
  done
  timed_fn download-swissprot fetch https://ftp.ncbi.nlm.nih.gov/blast/db/FASTA/swissprot.gz \
    "$ROOT/references/swissprot.gz" md5 1f885c2bba9d35231f920b71f7fa3b36
  n=$(zcat swissprot.gz | grep -c '^>')
  echo "swissprot sequences $n" | tee swissprot.count
  date -u +%FT%TZ > swissprot.downloaded_utc
fi
