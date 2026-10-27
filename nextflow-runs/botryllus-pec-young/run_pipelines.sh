#!/bin/bash
# Runs inside the Nextflow head job (scripts/nf-head): nf-core/rnaseq, then
# nf-core/differentialabundance on its gene counts. Each pipeline has its own launch folder,
# so `-resume` in one never picks up the other's last session.
# Usage: run_pipelines.sh <repo run dir> <output root> <nextflow binary>
set -euo pipefail
REPO=$1; OUT=$2; NEXTFLOW=$3

mkdir -p "$OUT/rnaseq-launch" "$OUT/differentialabundance-launch"

cd "$OUT/rnaseq-launch"
"$NEXTFLOW" run nf-core/rnaseq -r 3.27.0 -resume \
  -c "$REPO/sherlock.config" -params-file "$REPO/params-rnaseq.yaml" \
  --input "$REPO/samplesheet.csv" --outdir "$OUT/rnaseq"

COUNTS=$OUT/rnaseq/star_salmon/salmon.merged.gene_counts.tsv
LENGTHS=$OUT/rnaseq/star_salmon/salmon.merged.gene_lengths.tsv
[ -s "$COUNTS" ] || { echo "nf-core/rnaseq finished without $COUNTS" >&2; exit 3; }

cd "$OUT/differentialabundance-launch"
"$NEXTFLOW" run nf-core/differentialabundance -r 2.0.0 -profile rnaseq -resume \
  -c "$REPO/sherlock.config" -params-file "$REPO/params-differentialabundance.yaml" \
  --input "$REPO/samples.csv" --contrasts "$REPO/contrasts.csv" \
  --matrix "$COUNTS" --feature_length_matrix "$LENGTHS" \
  --outdir "$OUT/differentialabundance"
