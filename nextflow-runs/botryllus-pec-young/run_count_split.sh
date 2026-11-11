#!/bin/bash
# Runs inside the Nextflow head job (scripts/nf-head), for `make run-count-split`:
#   1. nf-core/rnaseq on the paper's 38 libraries against the paper's genome and gene models
#      (only if <genomes> names paper_genome)
#   2. count_split.nf: featureCounts on the alignments of each genome, then the fits
# Each has its own launch folder, so `-resume` in one never picks up the other's session.
# Usage: run_count_split.sh <repo run dir> <output root> <nextflow binary> <genomes>
set -euo pipefail
REPO=$1; OUT=$2; NEXTFLOW=$3; GENOMES=$4

[ -s "$OUT/rnaseq/star_salmon/salmon.merged.gene_counts.tsv" ] || { echo "missing $OUT/rnaseq; run make run first" >&2; exit 3; }
[ -s "$OUT/size-factors/paper_counts.tsv" ] || { echo "missing $OUT/size-factors/paper_counts.tsv (each-alone-checks/export_paper_counts.py)" >&2; exit 3; }

case ",$GENOMES," in *,paper_genome,*)
  mkdir -p "$OUT/rnaseq-paper-genome-launch"
  cd "$OUT/rnaseq-paper-genome-launch"
  "$NEXTFLOW" run nf-core/rnaseq -r 3.27.0 -resume \
    -c "$REPO/sherlock.config" -params-file "$REPO/params-rnaseq-paper-genome.yaml" \
    --input "$REPO/samplesheet_paper.csv" --outdir "$OUT/rnaseq-paper-genome"
  ;;
esac

mkdir -p "$OUT/count-split-launch"
cd "$OUT/count-split-launch"
"$NEXTFLOW" run "$REPO/count_split.nf" -resume -c "$REPO/sherlock.config" \
  --out "$OUT" --repo "$REPO" --genomes "$GENOMES"
