#!/bin/bash
#SBATCH -J geo2kmers-run
#SBATCH -p normal,hns
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 00:40:00
# The test run in the README: the default workflow, then the same with the species' own proteome.
# Submit from an empty folder with this repository's folder in CODE, an environment built from
# environment.yml in PYTHON, the proteome in PROTEOME, and NCBI_EMAIL set.
set -uo pipefail
: "${CODE:?}" "${PYTHON:?}" "${PROTEOME:?}" "${NCBI_EMAIL:?}"
ml load biology nextflow
export NXF_OPTS="-Xms512m -Xmx2g"
top=$PWD
(cd "$CODE" && GEO2KMERS_TEST_NEXTFLOW=1 "$PYTHON" -m pytest tests -q -rs -p no:cacheprovider | tail -5)
run() {
  mkdir -p "$top/$1" && cd "$top/$1" || exit 1
  local out=$top/$1/results/GSE2183 start status
  start=$(date +%s)
  nextflow run "$CODE/main.nf" -profile small --gse GSE2183 --outdir "$out" --python_fetch "$PYTHON" \
    --python_sequences "$PYTHON" --python_sketch "$PYTHON" "${@:2}" | tail -2
  status=${PIPESTATUS[0]}
  echo "wall_seconds $(($(date +%s) - start)) exit $status" | tee "$out/wall_time.txt"
  "$PYTHON" "$CODE/summarise.py" "$out/sketches" "$out/sequences/probes.fasta" \
    "$out/sequences/probe_sequence_source.tsv" "$out"
  command cp geo2kmers.trace.txt "$out/"
}
run default
run proteome --proteome "$PROTEOME"
