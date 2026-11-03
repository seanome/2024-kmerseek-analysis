#!/bin/bash
#SBATCH --job-name=nb361-human-download
#SBATCH --account=ayeletv
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x.%A_%a.out
# Download one run of a human GEO series from ENA (added 2026-11-03 for GSE120862, Olga:
# "download GSE120862 (84 runs) with MD5 checks"). Same transfer rules as 21_sra_download.sbatch:
# resumable curl -C -, --fail so an HTTP error page is never saved as a FASTQ, a speed floor so a
# stalled transfer restarts, and the file is kept only if its MD5 matches ENA's fastq_md5.
# Writes <out_dir>/<run>.fastq.gz, the same name the human Nextflow pipeline's download step
# stores, so the pipeline finds these files and does not download them again.
#
#   SERIES=GSE120862 RUNS_TSV=.../paper_accession_runs.tsv OUT_DIR=$S/human/reads/GSE120862 \
#     sbatch --export=ALL --array=0-83%8 28_human_download.sbatch
# Array index N = the N-th row of RUNS_TSV whose first column is SERIES.
set -euo pipefail
: "${SERIES:?}" "${RUNS_TSV:?}" "${OUT_DIR:?}"
mkdir -p "$OUT_DIR"
line=$(awk -F'\t' -v s="$SERIES" '$1 == s' "$RUNS_TSV" | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")
[ -n "$line" ] || { echo "no row $SLURM_ARRAY_TASK_ID for $SERIES"; exit 2; }
run=$(cut -f2 <<< "$line"); ftp=$(cut -f11 <<< "$line"); md5=$(cut -f12 <<< "$line")
case "$ftp" in *";"*) echo "$run is paired; this script handles single-end runs only"; exit 2;; esac
out=$OUT_DIR/$run.fastq.gz
if [ -e "$out" ]; then echo "$run already done"; exit 0; fi
for attempt in $(seq 1 20); do
  # no --retry: curl restarts a retried transfer from byte 0 (seen 2026-11-03, a 1.68 GB part fell
  # back to 317 MB); the loop below resumes with -C - instead
  if curl -sS -L --fail -C - --speed-limit 100000 --speed-time 60 \
       -o "$out.part" "https://$ftp"; then
    break
  fi
  echo "attempt $attempt stalled or failed; resuming"
  sleep 20
done
got=$(md5sum "$out.part" | cut -d' ' -f1)
[ "$got" = "$md5" ] || { echo "MD5 mismatch for $run: $got vs $md5"; command rm -f "$out.part"; exit 1; }
mv "$out.part" "$out"
echo "$run ok $got"
