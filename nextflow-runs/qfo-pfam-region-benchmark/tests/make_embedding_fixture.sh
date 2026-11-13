#!/usr/bin/env bash
# Build tests/fixtures/embedding/ for the TM-Vec and PLMSearch steps. Run by hand, never
# by CI.
#
# Neither tool runs on a laptop (both need a GPU and multi-GB model weights), so these
# files are HAND-BUILT in each tool's output layout from the committed MMseqs2 fixture
# (human fixture queries against the yeast fixture proteome):
#   tmvec.tabular.tsv      TM-Vec tabular layout (query_id, rank, database_id, tm-score);
#                          the "tm-score" column holds MMseqs2 percent identity / 100
#   tmvec.alignments.txt   DeepBLAST layout (name, gapped residues; query line first),
#                          holding the MMseqs2 alignment of each best pair
#   plmsearch.results.tsv  PLMSearch layout (query, target, score), the top three
#                          MMseqs2 hits per query with bit score as the score
# They test the parsers, the wiring and the scoring path, not the tools' results.
set -euo pipefail

HERE=$(cd "$(dirname "$0")/.." && pwd)
FIX="$HERE/tests/fixtures"
OUT="$FIX/embedding"
ENVS="${CONDA_ENVS:-$HOME/anaconda3/envs}"
PY="$ENVS/2025-kmerseek-analysis/bin/python3"
TMP=$(mktemp -d)
trap 'command rm -rf "$TMP"' EXIT
mkdir -p "$OUT"

"$PY" - "$FIX/annotator/mmseqs2_besthit.s7.5.hits.tsv.gz" "$OUT" <<'PYEOF'
import sys
from pathlib import Path
import polars as pl
sys.path.insert(0, str(Path(sys.argv[1]).parents[3] / "bin"))
import annotator_best_hit as abh

hits = abh.read_hits(Path(sys.argv[1]))
out = Path(sys.argv[2])
best = abh.best_hits(hits)
tab = ["query_id\trank\tdatabase_id\ttm-score"]
aln = []
for r in best.iter_rows(named=True):
    tab.append(f"{r['query']}\t0\t{r['target']}\t{r['pident'] / 100:.4f}")
    aln += [f"{r['query']}  {r['qaln']}", f"{r['target']}  {r['taln']}"]
(out / "tmvec.tabular.tsv").write_text("\n".join(tab) + "\n")
(out / "tmvec.alignments.txt").write_text("\n".join(aln) + "\n")
top = (hits.sort(["query", "bits"], descending=[False, True])
           .group_by("query", maintain_order=True).head(3))
(out / "plmsearch.results.tsv").write_text("".join(
    f"{r['query']}\t{r['target']}\t{r['bits']}\n" for r in top.iter_rows(named=True)))
PYEOF

"$PY" "$HERE/bin/embedding_calls.py" short-headers --fasta "$FIX/qfo/human.fasta" \
  --out "$TMP/query.short.fasta"
sed 's/^>sp|\([^|]*\)|.*/>\1/' "$FIX/qfo/yeast.fasta" > "$TMP/yeast.short.fasta"
MAP="$FIX/annotations/yeast_pfam_domains.parquet"

"$PY" "$HERE/bin/embedding_calls.py" tmvec --tabular "$OUT/tmvec.tabular.tsv" \
  --alignments "$OUT/tmvec.alignments.txt" --targets "$TMP/yeast.short.fasta" \
  --queries "$TMP/query.short.fasta" --domain-map "$MAP" --setting cath_deepblast \
  --species yeast --out "$OUT/tmvec.cath_deepblast.yeast.regions.tsv.gz" \
  --qc-out "$OUT/tmvec.cath_deepblast.yeast.calls_qc.json" 2> /dev/null
"$PY" "$HERE/bin/embedding_calls.py" plmsearch --results "$OUT/plmsearch.results.tsv" \
  --queries "$TMP/query.short.fasta" --domain-map "$MAP" --setting ss_predictor \
  --species yeast --out "$OUT/plmsearch.ss_predictor.yeast.regions.tsv.gz" \
  --qc-out "$OUT/plmsearch.ss_predictor.yeast.calls_qc.json" 2> /dev/null

"$PY" "$HERE/bin/build_domain_truth.py" --annotations "$FIX/annotations" \
  --truth-out "$TMP/human_domain_truth.parquet" --map-outdir "$TMP" \
  --summary-out "$TMP/truth_summary.json" > /dev/null
for pair in tmvec:cath_deepblast plmsearch:ss_predictor; do
  tool=${pair%%:*}; setting=${pair#*:}
  stem="pfam.$tool.$setting.yeast"
  "$PY" "$HERE/bin/evaluate_domain_calls.py" \
    --regions "$OUT/$tool.$setting.yeast.regions.tsv.gz" \
    --tool "$tool" --variant "$setting" --species yeast \
    --truth "$TMP/human_domain_truth.parquet" --direct-annotation \
    --calls-out "$OUT/$stem.calls.parquet" --metrics-out "$OUT/$stem.metrics.parquet" \
    --curve-out "$OUT/$stem.curve.parquet" > /dev/null 2>&1
done
ls -la "$OUT"
