#!/bin/bash
# Runs inside the Nextflow head job: nf-core/rnaseq with STAR + RSEM, then
# nf-core/differentialabundance on RSEM's gene counts, rounded to whole numbers for DESeq2.
# Usage: run_pipelines.sh <this folder> <output root> <nextflow binary> <python>
set -euo pipefail
REPO=$1; OUT=$2; NEXTFLOW=$3; PY=$4

mkdir -p "$OUT/rnaseq-launch" "$OUT/differentialabundance-launch"

cd "$OUT/rnaseq-launch"
"$NEXTFLOW" run nf-core/rnaseq -r 3.27.0 -resume \
  -c "$REPO/sherlock.config" -params-file "$REPO/params-rnaseq.yaml" \
  --input "$OUT/design/samplesheet.csv" --outdir "$OUT/rnaseq"

COUNTS=$OUT/rnaseq/star_rsem/rsem.merged.gene_counts.tsv
[ -s "$COUNTS" ] || { echo "nf-core/rnaseq finished without $COUNTS" >&2; exit 3; }
"$PY" - "$COUNTS" "$OUT/design/rsem_counts_whole.tsv" <<'PYEOF'
import sys

import pandas as pd

d = pd.read_csv(sys.argv[1], sep="\t").drop(columns=["gene_name"]).set_index("gene_id")
assert len(d.columns) == 47, len(d.columns)
d.round().astype(int).to_csv(sys.argv[2], sep="\t")
PYEOF

cd "$OUT/differentialabundance-launch"
"$NEXTFLOW" run nf-core/differentialabundance -r 2.0.0 -profile rnaseq -resume \
  -c "$REPO/sherlock.config" -params-file "$REPO/params-differentialabundance.yaml" \
  --input "$OUT/design/samples.csv" --contrasts "$OUT/design/contrasts.csv" \
  --matrix "$OUT/design/rsem_counts_whole.tsv" --outdir "$OUT/differentialabundance"
