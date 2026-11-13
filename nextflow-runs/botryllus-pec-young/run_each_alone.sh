#!/bin/bash
# Runs inside the Nextflow head job (scripts/nf-head): nf-core/differentialabundance twice on
# the gene counts of the finished nf-core/rnaseq run, each comparison fitted on its own
# libraries with the design ~ group and no colony term.
#   differentialabundance-paper         the paper's four comparisons on its 38 libraries,
#                                       DESeq2's default size factors
#   differentialabundance-each-window   the six windows on all 52 libraries, poscounts
# Each has its own launch folder, so `-resume` in one never picks up the other's session.
# Usage: run_each_alone.sh <repo run dir> <output root> <nextflow binary>
set -euo pipefail
REPO=$1; OUT=$2; NEXTFLOW=$3

COUNTS=$OUT/rnaseq/star_salmon/salmon.merged.gene_counts.tsv
LENGTHS=$OUT/rnaseq/star_salmon/salmon.merged.gene_lengths.tsv
[ -s "$COUNTS" ] || { echo "missing $COUNTS; run make run first" >&2; exit 3; }

mkdir -p "$OUT/differentialabundance-paper-launch" "$OUT/differentialabundance-each-window-launch"

# DESeq2's default size factors (ratio) use only genes counted in every library. 392 genes
# are counted in all 38 of the paper's libraries, so they work here, and they are what the
# paper used: refitting its counts with them gives back its p-values.
cd "$OUT/differentialabundance-paper-launch"
"$NEXTFLOW" run nf-core/differentialabundance -r 2.0.0 -profile rnaseq -resume \
  -c "$REPO/sherlock.config" -params-file "$REPO/params-differentialabundance.yaml" \
  --deseq2_sf_type ratio --differential_subset_to_contrast_samples true \
  --input "$REPO/samples_paper.csv" --contrasts "$REPO/contrasts_paper.csv" \
  --matrix "$COUNTS" --feature_length_matrix "$LENGTHS" \
  --outdir "$OUT/differentialabundance-paper"

# vst stops unless at least nsub genes have a mean normalised count above 5. Fitted alone,
# 21-28 d has 14 such genes, 49-52 d has 992 and 13-17 d has 490, so the default of 1000
# stops the run. nsub only changes the transformed tables and the report's plots, not the
# fold changes or p-values.
cd "$OUT/differentialabundance-each-window-launch"
"$NEXTFLOW" run nf-core/differentialabundance -r 2.0.0 -profile rnaseq -resume \
  -c "$REPO/sherlock.config" -params-file "$REPO/params-differentialabundance.yaml" \
  --deseq2_vst_nsub 10 --differential_subset_to_contrast_samples true \
  --input "$REPO/samples.csv" --contrasts "$REPO/contrasts_each_window.csv" \
  --matrix "$COUNTS" --feature_length_matrix "$LENGTHS" \
  --outdir "$OUT/differentialabundance-each-window"
