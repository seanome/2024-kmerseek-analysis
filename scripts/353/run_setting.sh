#!/bin/bash
# Notebook 353: build one setting's index of the evidence collection, make its queries,
# search them and score the hits. Runs inside a SLURM job on Sherlock (never the laptop).
#
#   run_setting.sh <split: dev|held_out> <alphabet> <k> <out root> [step] [chunk]
#
# step all (default; the development half): index, queries, search, score in one job.
# The held-out half is 20 times bigger and kmerseek's CSV output runs on about one core, so
# it is split into steps that held_out.sbatch chains with SLURM dependencies:
#   prep           index (if not built), queries, segmasker, queries split into N_CHUNKS
#   search <i>     search chunk i (one array task each)
#   score          join the chunks' regions and score
#
# dev:      1,000 development records, 12 substitution levels, 2 dipeptide shuffles
# held_out: every held-out record, 12 levels, 20 composition and 20 dipeptide shuffles,
#           plus segmasker on every window (the low-complexity flag)
# Writes <out root>/index/<alphabet>.k<k>/ and <out root>/<split>/<alphabet>.k<k>/.
set -euo pipefail

split=$1 alphabet=$2 k=$3 root=$4 step=${5:-all} chunk=${6:-}
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd "$here/../.." && pwd)
img=${NB353_IMAGE:-$SCRATCH/apptainer-cache/docker.io-olgabot-kmerseek-2026-10-18-nb344-0a22970.img}
run() { apptainer exec --bind /scratch "$img" "$@"; }
export RAYON_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

records=$repo/data/kill_tests/short_stretch/records.tsv
fasta=$repo/data/kill_tests/short_stretch/records.fasta
penalty=2 xdrop=8 max_e=10
N_CHUNKS=${N_CHUNKS:-24}

idx=$root/index/$alphabet.k$k
out=$root/$split/$alphabet.k$k
mkdir -p "$idx" "$out"

build_index() {
  [ -s "$idx/done" ] && return 0
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
}

make_queries() {
  case $split in
    dev)      qargs="--n-records 1000 --comp-shuffles 0 --dip-shuffles 2" ;;
    held_out) qargs="--n-records 0 --comp-shuffles 20 --dip-shuffles 20" ;;
  esac
  run python3 "$here/make_queries.py" --records "$records" --split "$split" --alphabet "$alphabet" \
    $qargs --out-prefix "$out/queries"
  if [ "$split" = held_out ]; then
    run segmasker -in "$out/queries.fasta" -infmt fasta -outfmt interval -out "$out/queries.seg.txt"
  fi
}

# search <query fasta> <tag>: regions at E <= max_e into <tag>.regions.tsv.gz
search() {
  local q=$1 tag=$2
  set +e
  /usr/bin/time -v -o "$out/$tag.search.time" \
    apptainer exec --bind /scratch "$img" kmerseek search --query "$q" --target "$idx/collection.db" \
      --alphabet "$alphabet" --ksize "$k" \
      --extend-mismatch-penalty $penalty --extend-xdrop $xdrop 2> "$out/$tag.search.log" \
    | awk -v max_e=$max_e -v summary="$out/$tag.filter_summary.tsv" -f "$here/filter_regions.awk" \
    | gzip > "$out/$tag.regions.tsv.gz"
  local status=("${PIPESTATUS[@]}")
  set -e
  if [ "${status[0]}" -ne 0 ] || [ "${status[1]}" -ne 0 ] || [ "${status[2]}" -ne 0 ]; then
    echo "353: search pipeline failed on $q, exit codes ${status[*]}" >&2
    tail -20 "$out/$tag.search.log" >&2
    exit 1
  fi
}

score() {
  run python3 "$here/score.py" --split "$split" --records "$records" --queries "$out/queries.tsv" \
    --regions "$out/regions.tsv.gz" --out-dir "$out"
  date > "$out/done"
}

case $step in
  all)
    build_index
    make_queries
    search "$out/queries.fasta" all
    # The development run named these without the tag; keep those names.
    mv "$out/all.regions.tsv.gz" "$out/regions.tsv.gz"
    mv "$out/all.filter_summary.tsv" "$out/filter_summary.tsv"
    mv "$out/all.search.time" "$out/search.time"
    mv "$out/all.search.log" "$out/search.log"
    score
    ;;
  prep)
    build_index
    make_queries
    mkdir -p "$out/chunks"
    # Queries are one header line and one sequence line each; deal them out in order.
    n=$(grep -c '^>' "$out/queries.fasta")
    per=$(( (n + N_CHUNKS - 1) / N_CHUNKS ))
    awk -v per=$per -v dir="$out/chunks" '/^>/{c=int(i++/per)} {printf "%s\n", $0 > sprintf("%s/chunk_%02d.fasta", dir, c)}' "$out/queries.fasta"
    ls "$out/chunks" | wc -l > "$out/chunks/n_chunks"
    ;;
  search)
    c=$(printf '%02d' "$chunk")
    search "$out/chunks/chunk_$c.fasta" "chunks/chunk_$c"
    ;;
  score)
    n=$(cat "$out/chunks/n_chunks")
    for i in $(seq -f '%02g' 0 $((n - 1))); do
      test -s "$out/chunks/chunk_$i.regions.tsv.gz" || { echo "353: chunk $i has no regions file" >&2; exit 1; }
    done
    # One header, then every chunk's rows.
    # (head closes the pipe early; under pipefail that would stop the script, so || true)
    header=$(zcat "$out/chunks/chunk_00.regions.tsv.gz" | head -1 || true)
    { echo "$header"
      for i in $(seq -f '%02g' 0 $((n - 1))); do zcat "$out/chunks/chunk_$i.regions.tsv.gz" | tail -n +2; done
    } | gzip > "$out/regions.tsv.gz"
    awk -F'\t' 'FNR==1{next} {a+=$1; b+=$2; c+=$3; d+=$4} END{print "rows_in\trows_kept\trows_no_evalue\tqueries_with_rows"; print a"\t"b"\t"c"\t"d}' \
      "$out"/chunks/chunk_*.filter_summary.tsv > "$out/filter_summary.tsv"
    score
    ;;
esac
