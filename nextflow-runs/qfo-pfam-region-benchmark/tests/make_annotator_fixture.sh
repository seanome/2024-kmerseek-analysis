#!/usr/bin/env bash
# Build tests/fixtures/annotator/ from the committed CI fixture (60 human queries, 120
# yeast proteins) with the real tools. Run by hand when the fixture needs regenerating,
# never by CI: CI and `make stub` read what this commits.
#
# The yeast proteome stands in for "Swiss-Prot minus Mammalia" (no mammal in it), so
# every step of annotator_tools.nf runs on real tool output:
#   searches   DIAMOND --sensitive, --ultra-sensitive; MMseqs2 -s 7.5, one round
#   targets    annotator_best_hit.py targets
#   domains    hmmscan --cut_ga against Pfam-A
#   place      annotator_best_hit.py place
#   score      evaluate_domain_calls.py --direct-annotation
#
# Local tool versions (2026-10-16): DIAMOND 2.2.6 (the pipeline pins 2.2.8), MMseqs2
# 18.8cc5c (the pipeline pins the image main.nf uses), HMMER from the `hmmer` env, Pfam-A
# from ~/data/pfam. The fixture tests the code path, not the tools' numbers.
#
# Usage: bash tests/make_annotator_fixture.sh   (from the pipeline folder)
set -euo pipefail

HERE=$(cd "$(dirname "$0")/.." && pwd)
FIX="$HERE/tests/fixtures"
OUT="$FIX/annotator"
ENVS="${CONDA_ENVS:-$HOME/anaconda3/envs}"
PY="$ENVS/2025-kmerseek-analysis/bin/python3"
DIAMOND="$ENVS/diamond/bin/diamond"
MMSEQS="$ENVS/orthofinder/bin/mmseqs"
HMMSCAN="$ENVS/hmmer/bin/hmmscan"
PFAM="${PFAM_HMM:-$HOME/data/pfam/Pfam-A.hmm}"
SPECIES=swissprot_minus_mammalia

TMP=$(mktemp -d)
trap 'command rm -rf "$TMP"' EXIT
mkdir -p "$OUT"

# Plain accessions on the reference side, as build_clade_excluded_reference.py writes them.
sed 's/^>sp|\([^|]*\)|.*/>\1/' "$FIX/qfo/yeast.fasta" > "$TMP/ref.fasta"

FMT_D="qseqid sseqid qstart qend sstart send bitscore evalue pident qseq_gapped sseq_gapped"
"$DIAMOND" makedb --in "$TMP/ref.fasta" -d "$TMP/ref" --quiet
for s in sensitive ultra-sensitive; do
  "$DIAMOND" blastp -q "$FIX/qfo/human.fasta" -d "$TMP/ref" "--$s" --quiet \
    -o "$TMP/hits.tsv" --outfmt 6 $FMT_D
  gzip -nc "$TMP/hits.tsv" > "$OUT/diamond_besthit.$s.hits.tsv.gz"
done

"$MMSEQS" createdb "$FIX/qfo/human.fasta" "$TMP/qdb" -v 1
"$MMSEQS" createdb "$TMP/ref.fasta" "$TMP/tdb" -v 1
"$MMSEQS" search "$TMP/qdb" "$TMP/tdb" "$TMP/res" "$TMP/mtmp" -s 7.5 -a -v 1
"$MMSEQS" convertalis "$TMP/qdb" "$TMP/tdb" "$TMP/res" "$TMP/m.tsv" -v 1 \
  --format-output "query,target,qstart,qend,tstart,tend,bits,evalue,pident,qaln,taln"
gzip -nc "$TMP/m.tsv" > "$OUT/mmseqs2_besthit.s7.5.hits.tsv.gz"

"$PY" "$HERE/bin/annotator_best_hit.py" targets \
  --hits "$OUT"/*.hits.tsv.gz --reference "$TMP/ref.fasta" --out "$OUT/hit_targets.fasta"
"$HMMSCAN" --domtblout "$TMP/dom.txt" -o /dev/null --noali --cut_ga --cpu 4 \
  "$PFAM" "$OUT/hit_targets.fasta"
grep -v '^#' "$TMP/dom.txt" \
  | awk 'NF >= 22 {print $4 "\t" $2 "\t" $20 "\t" $21 "\t" $14 "\t" $13}' \
  | gzip -nc > "$OUT/hit_target_domains.tsv.gz"

"$PY" "$HERE/bin/build_domain_truth.py" --annotations "$FIX/annotations" \
  --truth-out "$TMP/human_domain_truth.parquet" --map-outdir "$TMP" \
  --summary-out "$TMP/truth_summary.json" > /dev/null

for pair in diamond_besthit:sensitive diamond_besthit:ultra-sensitive mmseqs2_besthit:s7.5; do
  tool=${pair%%:*}; setting=${pair#*:}
  stem="$tool.$setting"
  "$PY" "$HERE/bin/annotator_best_hit.py" place \
    --hits "$OUT/$stem.hits.tsv.gz" --target-domains "$OUT/hit_target_domains.tsv.gz" \
    --queries "$FIX/qfo/human.fasta" --tool "$tool" --setting "$setting" \
    --out "$OUT/$stem.regions.tsv.gz" --qc-out "$OUT/$stem.place_qc.json" 2> /dev/null
  "$PY" "$HERE/bin/evaluate_domain_calls.py" --regions "$OUT/$stem.regions.tsv.gz" \
    --tool "$tool" --variant "$setting" --species "$SPECIES" \
    --truth "$TMP/human_domain_truth.parquet" --direct-annotation \
    --calls-out "$OUT/pfam.$stem.$SPECIES.calls.parquet" \
    --metrics-out "$OUT/pfam.$stem.$SPECIES.metrics.parquet" \
    --curve-out "$OUT/pfam.$stem.$SPECIES.curve.parquet" > /dev/null 2>&1
done
ls -la "$OUT"
