#!/bin/bash
# Runs inside the Nextflow head job: nf-core/rnaseq with STAR + RSEM, then
# nf-core/differentialabundance once per contrast on RSEM's gene counts, rounded to whole numbers
# for DESeq2. Smallest contrast first. A failed contrast does not stop the others.
# Usage: run_pipelines.sh <this folder> <output root> <nextflow binary> <python>
set -euo pipefail
REPO=$1; OUT=$2; NEXTFLOW=$3; PY=$4

mkdir -p "$OUT/rnaseq-launch"

cd "$OUT/rnaseq-launch"
"$NEXTFLOW" run nf-core/rnaseq -r 3.27.0 -resume \
  -c "$REPO/sherlock.config" -params-file "$REPO/params-rnaseq.yaml" \
  --input "$OUT/design/samplesheet.csv" --outdir "$OUT/rnaseq"

COUNTS=$OUT/rnaseq/star_rsem/rsem.merged.gene_counts.tsv
[ -s "$COUNTS" ] || { echo "nf-core/rnaseq finished without $COUNTS" >&2; exit 3; }
"$PY" - "$COUNTS" "$OUT/design" <<'PYEOF'
import sys
from pathlib import Path

import pandas as pd

d = pd.read_csv(sys.argv[1], sep="\t").drop(columns=["gene_name"]).set_index("gene_id")
assert len(d.columns) == 47, len(d.columns)
d = d.round().astype(int)
for sheet in Path(sys.argv[2]).glob("*/samples.csv"):
    d[pd.read_csv(sheet)["sample"].tolist()].to_csv(sheet.parent / "rsem_counts_whole.tsv", sep="\t")
PYEOF

failed=""
for c in old_24h old_41_93d young_132d young_13_49d young_2h young_24h; do
  mkdir -p "$OUT/differentialabundance-launch/$c"
  cd "$OUT/differentialabundance-launch/$c"
  "$NEXTFLOW" run nf-core/differentialabundance -r 2.0.0 -profile rnaseq -resume \
    -c "$REPO/sherlock.config" -params-file "$REPO/params-differentialabundance.yaml" \
    --input "$OUT/design/$c/samples.csv" --contrasts "$OUT/design/$c/contrasts.csv" \
    --matrix "$OUT/design/$c/rsem_counts_whole.tsv" --outdir "$OUT/differentialabundance/$c" || failed="$failed $c"
done
[ -z "$failed" ] || { echo "nf-core/differentialabundance failed for:$failed" >&2; exit 4; }
