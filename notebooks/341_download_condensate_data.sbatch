#!/bin/bash
#SBATCH --job-name=341-download
#SBATCH --account=ayeletv
#SBATCH --partition=hns
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=12:00:00
#SBATCH --output=341-download-%j.out
#
# Notebook 341: download every open dataset of the condensate-exon experiment to $SCRATCH.
# The list, with each file's URL, md5 and size, is data/341_condensate_exons/download_manifest.tsv
# (57 files, 294.9 GB):
#
#   tanaka2025_PRJDB19918   224.2 GB  nanopore and Illumina reads, iPSC motor neurons with TDP-43
#                                     shRNA or scramble shRNA, untreated neurons and iPSCs (ENA)
#   yang2023_PRJNA947248     70.3 GB  PacBio and Illumina reads, cerebral organoids (ENA)
#   xu2026_zenodo16897791     0.4 GB  isoform GTF, FASTA, SQANTI3 classification and counts,
#                                     iPSC to NGN2 neurons (Zenodo record 16897791)
#
# Each file is downloaded with resume (curl -C -), retried, dropped and restarted when it stalls
# below 100 kB/s for two minutes, and then checked against its md5. A file whose md5 matches gets
# a .md5ok marker and is skipped on a rerun, so a job that runs out of time is resumed by
# submitting it again. Four files download at once.
#
# Submit from a checkout of olgabot/condensate-exons-341 on Sherlock. First time only:
#
#   git clone --branch olgabot/condensate-exons-341 https://github.com/seanome/2024-kmerseek-analysis $SCRATCH/2024-kmerseek-analysis-341
#
# Then:
#
#   cd $SCRATCH/2024-kmerseek-analysis-341 && git pull && sbatch notebooks/341_download_condensate_data.sbatch
#
# Sizing: no earlier download from ENA to Sherlock is recorded, so the time is from bandwidth.
# At 20 MB/s per file and four at once, 295 GB takes about an hour; at 2 MB/s per file it takes
# about 10 h, inside the 12 h limit. md5 of a 20 GB file takes about a minute on one core.
# curl and md5sum use little memory; 4 G is generous. Read the real numbers after the run with
# `sacct -j <jobid> -o JobID,MaxRSS,Elapsed,State`.
set -uo pipefail

MANIFEST=${CONDENSATE_341_MANIFEST:-data/341_condensate_exons/download_manifest.tsv}
OUT=${CONDENSATE_341_DATA:-$SCRATCH/condensate_exons_341}
mkdir -p "$OUT"
echo "writing to $OUT; $(df -h "$OUT" | tail -1)"

download_one() {
    local dataset=$1 url=$2 md5=$3 bytes=$4
    local dir="$OUT/$dataset" name
    name=$(basename "${url%%\?*}")
    mkdir -p "$dir"
    local file="$dir/$name"
    if [[ -f "$file.md5ok" ]]; then
        echo "skip $dataset/$name (md5 already checked)"
        return 0
    fi
    local attempt
    for attempt in 1 2 3 4 5 6; do
        curl -fsSL -C - --retry 5 --retry-delay 30 --speed-limit 100000 --speed-time 120 -o "$file" "$url"
        local status=$?
        if [[ $status -eq 0 ]]; then
            break
        fi
        echo "curl exit $status on $dataset/$name, attempt $attempt"
        sleep 60
    done
    local got
    got=$(md5sum "$file" | cut -d' ' -f1)
    if [[ "$got" == "$md5" ]]; then
        touch "$file.md5ok"
        echo "ok $dataset/$name ($bytes bytes)"
    else
        rm -f "$file"
        echo "MD5 MISMATCH $dataset/$name: expected $md5, got $got; deleted, resubmit to download it again"
        return 1
    fi
}
export -f download_one
export OUT

# dataset, url, md5 and bytes have no spaces, so four words per line go to xargs as four arguments.
tail -n +2 "$MANIFEST" | awk -F'\t' '{print $1, $5, $6, $7}' \
    | xargs -n 4 -P 4 bash -c 'download_one "$@"' _
pipeline_status=$?

n_files=$(($(wc -l < "$MANIFEST") - 1))
n_ok=$(find "$OUT" -name '*.md5ok' | wc -l)
echo "$n_ok of $n_files files downloaded and checked"
if [[ $n_ok -ne $n_files || $pipeline_status -ne 0 ]]; then
    echo "not complete: resubmit the same command to resume"
    exit 1
fi
