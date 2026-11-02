#!/bin/bash
#SBATCH --job-name=nb361-sra-download
#SBATCH --account=ayeletv
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x.%A_%a.out
# Download one human run (single FASTQ) named in runs.tsv from ENA. Resumable (curl -C -),
# retries, and a speed floor so a stalled transfer is restarted. The file is kept only if its
# MD5 matches ENA's. Array index N = the N-th run whose dataset is not PRJNA1219309.
set -euo pipefail
: "${PEC_ROOT:?}"
cd "$PEC_ROOT/sra_search"
mkdir -p reads
line=$(awk -F'\t' 'NR>1 && $3!="PRJNA1219309"' runs.tsv | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")
run=$(cut -f1 <<< "$line"); ftp=$(cut -f10 <<< "$line"); md5=$(cut -f11 <<< "$line")
out=reads/$run.fastq.gz
[ -e "$out.done" ] && { echo "$run already done"; exit 0; }
for attempt in $(seq 1 20); do
  # --fail: an HTTP error page must not be saved as the FASTQ (32 runs got the same error page
  # under 12 concurrent downloads on 2026-11-02; their MD5 check caught it)
  if curl -sS -L --fail -C - --retry 5 --retry-delay 10 --speed-limit 100000 --speed-time 60 \
       -o "$out.part" "https://$ftp"; then
    break
  fi
  echo "attempt $attempt stalled or failed; resuming"
  sleep 20
done
got=$(md5sum "$out.part" | cut -d' ' -f1)
[ "$got" = "$md5" ] || { echo "MD5 mismatch for $run: $got vs $md5"; command rm -f "$out.part"; exit 1; }
mv "$out.part" "$out"
touch "$out.done"
echo "$run ok"
