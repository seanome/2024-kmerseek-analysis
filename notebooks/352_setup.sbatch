#!/bin/bash
#SBATCH --job-name=352-setup
#SBATCH --account=ayeletv
#SBATCH --partition=hns
#SBATCH --constraint=NO_GPU
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=352-setup-%j.out
#
# Notebook 352, setup. Submitted by notebooks/352_submit_all.sh. Rules:
# notebooks/352_data/preregistered.yaml.
#
#   1. download the 200 UniParc active FASTA parts (release 2026_03) and check each
#      against the md5 in RELEASE.metalink, fetching a failed part again up to 3 times
#   2. pass 1: residues per bin of the subset hash, over every sequence
#   3. the hash cut-off of each size (1, 10, 20, 30, 50, 100 billion residues)
#   4. pass 2: each sequence into the slice of the smallest size that holds it, so every
#      size's subset is its slices put together, and smaller subsets sit inside larger ones
#   5. the 1,000 queries, from the 1 billion residue subset
#
# Sizing (predictions, not measurements): 248.7 GB of .gz (sum of the metalink sizes).
# EBI served about 1.9 MB/s per connection in notebook 299's download (2026-10-19), so 24
# connections at once take about 1.5 h. Each pass reads every part once with 32 processes;
# no earlier run reads UniParc, so this is a guess: under 2 h for both. 12 h leaves room
# for slow downloads. 64 G: pass 1 holds one 2^24-bin histogram (134 MB) and one part's
# hash bins (about 5.6 million numbers) per process. The real use is in
# `sacct -j <job id> -o JobID,MaxRSS,Elapsed`.
set -euo pipefail
START=$(date +%s)

REPO=${SLURM_SUBMIT_DIR:?submit from the repo root}
WORK=${NB352_WORK:-$SCRATCH/352-kmerseek-scale}
IMG=${NB352_IMAGE:-$SCRATCH/apptainer-cache/docker.io-olgabot-kmerseek-0.4.0-rc5.img}
BASE=https://ftp.uniprot.org/pub/databases/uniprot/current_release/uniparc/fasta/active
mkdir -p "$WORK"/{parts,histogram,slices,results}
cd "$REPO"

export CUDA_VISIBLE_DEVICES=""
PY=(apptainer exec --bind /scratch "$IMG" python3 "$REPO/scripts/kmerseek_scale_352.py")
record_budget() {
    "${PY[@]}" budget-add "$WORK/budget.tsv" setup "${SLURM_JOB_ID:-none}" "$START" "$(date +%s)" || true
}
trap record_budget EXIT

echo "host $(hostname); $(lscpu | grep 'Model name')"
"${PY[@]}" check-cpu

# 1. Download. The metalink committed in the repo is the one read on 2026-10-20; the job
#    fetches it again and stops if the release changed since, because the md5 list and
#    the preregistered source would no longer match.
curl -fsSL -o "$WORK/parts/RELEASE.metalink" "$BASE/RELEASE.metalink"
if ! cmp -s "$WORK/parts/RELEASE.metalink" notebooks/352_data/uniparc_active_RELEASE.metalink; then
    echo "UniParc RELEASE.metalink differs from the committed copy (new release?). Stopping."
    exit 1
fi
fetch() {  # fetch FILE BYTES: resume on a stall (under 100 kB/s for 60 s), up to 30 tries
    local f=$1 want=$2
    for _ in $(seq 1 30); do
        [ -f "$WORK/parts/$f" ] && [ "$(stat -c %s "$WORK/parts/$f")" -eq "$want" ] && return 0
        curl -sSL -C - --speed-limit 100000 --speed-time 60 -o "$WORK/parts/$f" "$BASE/$f" && return 0
        sleep 5
    done
    return 1
}
export -f fetch
export WORK BASE
# One line per part: file name and its size in bytes, from the metalink.
FILES=$(apptainer exec --bind /scratch "$IMG" python3 -c '
import sys; sys.path.insert(0, "scripts"); import kmerseek_scale_352 as k
for name, (size, _) in sorted(k.parse_metalink("notebooks/352_data/uniparc_active_RELEASE.metalink").items()):
    if name.endswith(".fasta.gz"): print(name, size)')
echo "$(echo "$FILES" | wc -l) parts to fetch"
for round in 1 2 3 4; do
    echo "$FILES" | xargs -P 24 -L 1 bash -c 'fetch "$0" "$1"' || true
    if "${PY[@]}" verify-parts "$WORK/parts/RELEASE.metalink" "$WORK/parts" "$WORK/results/parts_md5.tsv" --workers 32; then
        break
    fi
    [ "$round" -eq 4 ] && { echo "parts still fail the md5 check after 4 rounds"; exit 1; }
    # A part that fails its md5 is fetched again from byte 0.
    awk -F'\t' 'NR > 1 && $6 != "True" { print $1 }' "$WORK/results/parts_md5.tsv" \
        | while read -r f; do rm -f "$WORK/parts/$f"; done
done
DOWNLOADED=$(date -u '+%F')
echo "download finished $DOWNLOADED, $(du -sb "$WORK/parts" | cut -f1) bytes"

# 2-3. Pass 1 and the cut-offs.
"${PY[@]}" histogram "$WORK/parts" "$WORK/histogram" --workers 32
"${PY[@]}" cutoffs "$WORK/histogram/histogram.npy" "$WORK/results/cutoffs.json"

# 4-5. Pass 2 and the queries.
"${PY[@]}" split "$WORK/parts" "$WORK/results/cutoffs.json" "$WORK/slices" --workers 32
cp "$WORK/slices/subsets.json" "$WORK/results/subsets.json"
"${PY[@]}" pick-queries "$WORK/slices/candidates" "$WORK/results/queries.fasta" "$WORK/results/queries.tsv"

# What data/SOURCES.md needs: URL, date downloaded, size and checksum of every file.
{
    echo "downloaded_utc_date	$DOWNLOADED"
    echo "source	$BASE/"
    echo "metalink_sha256	$(sha256sum "$WORK/parts/RELEASE.metalink" | cut -d' ' -f1)"
    echo "parts	$(awk -F'\t' 'NR > 1' "$WORK/results/parts_md5.tsv" | wc -l)"
    echo "parts_bytes	$(awk -F'\t' 'NR > 1 { s += $2 } END { printf "%d", s }' "$WORK/results/parts_md5.tsv")"
    echo "parts_md5_all_ok	$(awk -F'\t' 'NR > 1 && $6 != "True"' "$WORK/results/parts_md5.tsv" | wc -l | sed 's/^0$/yes/')"
} > "$WORK/results/download_record.tsv"
cat "$WORK/results/download_record.tsv"
echo "setup done"
