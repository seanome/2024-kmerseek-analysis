#!/bin/bash
# Notebook 353: build one setting's index of the evidence collection, make its queries,
# search them and score the hits. Runs inside a SLURM job on Sherlock (never the laptop).
#
#   run_setting.sh <split: dev|held_out> <alphabet> <k> <out root>
#
# dev:      1,000 development records, 12 substitution levels, 2 dipeptide shuffles
# held_out: every held-out record, 12 levels, 20 composition and 20 dipeptide shuffles,
#           plus segmasker on every window (the low-complexity flag)
# Writes <out root>/index/<alphabet>.k<k>/ and <out root>/<split>/<alphabet>.k<k>/.
set -euo pipefail

split=$1 alphabet=$2 k=$3 root=$4
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd "$here/../.." && pwd)
img=${NB353_IMAGE:-$SCRATCH/apptainer-cache/docker.io-olgabot-kmerseek-2026-10-18-nb344-0a22970.img}
run() { apptainer exec --bind /scratch "$img" "$@"; }
export RAYON_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

records=$repo/data/kill_tests/short_stretch/records.tsv
fasta=$repo/data/kill_tests/short_stretch/records.fasta
penalty=2 xdrop=8 max_e=10

idx=$root/index/$alphabet.k$k
out=$root/$split/$alphabet.k$k
mkdir -p "$idx" "$out"

if [ ! -s "$idx/done" ]; then
  /bin/rm -rf "$idx/collection.db"
  {
    run kmerseek --version
    run cat /etc/kmerseek_git_sha
    sha256sum "$img" "$fasta"
  } > "$idx/index.log"
  /usr/bin/time -v -o "$idx/index.time" \
    apptainer exec --bind /scratch "$img" kmerseek index --input "$fasta" --output "$idx/collection.db" \
      --alphabet "$alphabet" --ksize "$k" \
      --extend-mismatch-penalty $penalty --extend-xdrop $xdrop 2>> "$idx/index.log"
  # A refused Karlin-Altschul fit still exits 0; record it, region_evalue then falls back
  # to the run E-value (region_evalue_source says which each row used).
  if grep -q 'Stored in the index for' "$idx/index.log"; then echo yes > "$idx/ka_fit_stored"; else echo no > "$idx/ka_fit_stored"; fi
  date > "$idx/done"
fi

case $split in
  dev)      qargs="--n-records 1000 --comp-shuffles 0 --dip-shuffles 2" ;;
  held_out) qargs="--n-records 0 --comp-shuffles 20 --dip-shuffles 20" ;;
esac
run python3 "$here/make_queries.py" --records "$records" --split "$split" --alphabet "$alphabet" \
  $qargs --out-prefix "$out/queries"

if [ "$split" = held_out ]; then
  run segmasker -in "$out/queries.fasta" -infmt fasta -outfmt interval -out "$out/queries.seg.txt"
fi

set +e
/usr/bin/time -v -o "$out/search.time" \
  apptainer exec --bind /scratch "$img" kmerseek search --query "$out/queries.fasta" --target "$idx/collection.db" \
    --alphabet "$alphabet" --ksize "$k" \
    --extend-mismatch-penalty $penalty --extend-xdrop $xdrop 2> "$out/search.log" \
  | awk -v max_e=$max_e -v summary="$out/filter_summary.tsv" -f "$here/filter_regions.awk" \
  | gzip > "$out/regions.tsv.gz"
status=("${PIPESTATUS[@]}")
set -e
if [ "${status[0]}" -ne 0 ] || [ "${status[1]}" -ne 0 ] || [ "${status[2]}" -ne 0 ]; then
  echo "353: search pipeline failed, exit codes ${status[*]}" >&2
  tail -20 "$out/search.log" >&2
  exit 1
fi

run python3 "$here/score.py" --split "$split" --records "$records" --queries "$out/queries.tsv" \
  --regions "$out/regions.tsv.gz" --out-dir "$out"
date > "$out/done"
