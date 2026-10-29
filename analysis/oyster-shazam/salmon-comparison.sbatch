#!/bin/bash
#SBATCH --job-name=oyster-salmon-DESeq2
#SBATCH --partition=normal,hns
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
set -euo pipefail
ml biology salmon/1.10.0
source "$HOME/oyster-mvp-code/config.sh"
OUT="$SCRATCH/oyster-quantifier-comparison-20261024"
mkdir -p "$OUT"
exec >"$OUT/run.log" 2>&1
CODE="$HOME/oyster-quantifier-comparison-code"
PY="$SHAZAM_ENV/bin/python"
salmon --version > "$OUT/salmon-version.txt"
sha256sum "$CODE/"* "$ROOT/references/rna.fna.gz" "$ROOT/references/feature_table.txt.gz" > "$OUT/source-hashes.txt"
# Same transcript-only reference as kallisto; no genomic decoys in either arm.
salmon index -t "$ROOT/references/rna.fna.gz" -i "$OUT/index" -p 8 --keepDuplicates
for sample in "${SAMPLES[@]}"; do
  test -s "$ROOT/trimmed/full/$sample.validated.json"
  sha256sum "$ROOT/trimmed/full/${sample}_val_1.fq.gz" "$ROOT/trimmed/full/${sample}_val_2.fq.gz" >> "$OUT/read-hashes.txt"
  /usr/bin/time -v salmon quant -i "$OUT/index" -l A -1 "$ROOT/trimmed/full/${sample}_val_1.fq.gz" \
    -2 "$ROOT/trimmed/full/${sample}_val_2.fq.gz" -p 8 --validateMappings -o "$OUT/quant/$sample"
  "$PY" "$CODE/salmon_to_kallisto.py" "$OUT/quant/$sample" "$OUT/compatible/$sample"
done
"$PY" "$CODE/aggregate_quantifier_genes.py" --feature-table "$ROOT/references/feature_table.txt.gz" \
 --quant-dir "$OUT/compatible" --samples "$ROOT/baseline/full/samples.tsv" --out "$OUT/genes"
# Run the same R DESeq2 model on both arms, separately from the earlier PyDESeq2 fit.
"$DEKUPL_ENV/bin/Rscript" "$CODE/compare_quantifiers.R" "$ROOT/baseline/full/gene_counts.tsv" \
 "$OUT/genes/gene_counts.tsv" "$ROOT/baseline/full/samples.tsv" "$OUT/comparison"
sha256sum "$OUT/comparison/"* > "$OUT/result-hashes.txt"
touch "$OUT/complete"
