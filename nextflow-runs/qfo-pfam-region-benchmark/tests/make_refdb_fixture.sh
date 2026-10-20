#!/usr/bin/env bash
# Build tests/fixtures/refdb/ for the InterProScan and eggNOG-mapper steps. Run by hand,
# never by CI.
#
# Neither tool runs on a laptop (InterProScan's data is 6.9 GB compressed, eggNOG's 12 GB),
# so these fixtures are HAND-BUILT in each tool's output layout from hmmscan --cut_ga on
# the 60 human CI fixture proteins:
#   interproscan.all_members.tsv.gz  InterProScan 5 TSV layout, analysis "Pfam", E-value
#                                    in column 9, plus the same rows relabelled "Gene3D"
#                                    so the Pfam filter has something to drop
#   eggnog_mapper.diamond.emapper.annotations
#                                    eggNOG-mapper 2.1 layout; PFAMs = the hmmscan family
#                                    NAMES of that query, seed score = best domain score
# They test the parsers, the Nextflow wiring and the scoring path, not the tools' results.
set -euo pipefail

HERE=$(cd "$(dirname "$0")/.." && pwd)
FIX="$HERE/tests/fixtures"
OUT="$FIX/refdb"
ENVS="${CONDA_ENVS:-$HOME/anaconda3/envs}"
PY="$ENVS/2025-kmerseek-analysis/bin/python3"
HMMSCAN="$ENVS/hmmer/bin/hmmscan"
PFAM="${PFAM_HMM:-$HOME/data/pfam/Pfam-A.hmm}"

TMP=$(mktemp -d)
trap 'command rm -rf "$TMP"' EXIT
mkdir -p "$OUT"

"$HMMSCAN" --domtblout "$TMP/dom.txt" -o /dev/null --noali --cut_ga --cpu 4 \
  "$PFAM" "$FIX/qfo/human.fasta"

"$PY" - "$TMP/dom.txt" "$OUT" <<'PYEOF'
import sys
from pathlib import Path

dom, out = Path(sys.argv[1]), Path(sys.argv[2])
rows = []
for line in dom.read_text().splitlines():
    if line.startswith("#"):
        continue
    f = line.split()
    # name acc ... query qacc qlen ... i-Evalue(13) score(14) ... env from/to (20, 21)
    rows.append(dict(name=f[0], acc=f[1], query=f[3], qlen=f[5], ievalue=f[12],
                     score=float(f[13]), start=f[19], end=f[20]))
ips = []
for analysis in ("Pfam", "Gene3D"):
    for r in rows:
        ips.append("\t".join([r["query"], "0" * 32, r["qlen"], analysis,
                              r["acc"] if analysis == "Pfam" else "G3DSA:1.10.10.10",
                              r["name"], r["start"], r["end"], r["ievalue"], "T",
                              "16-10-2026", "-", "-"]))
import gzip
with gzip.GzipFile(out / "interproscan.all_members.tsv.gz", "wb", mtime=0) as fh:
    fh.write(("\n".join(ips) + "\n").encode())

by_q = {}
for r in rows:
    q = by_q.setdefault(r["query"], {"names": [], "score": 0.0, "evalue": r["ievalue"]})
    if r["name"] not in q["names"]:
        q["names"].append(r["name"])
    q["score"] = max(q["score"], r["score"])
cols = ["query", "seed_ortholog", "evalue", "score", "eggNOG_OGs", "max_annot_lvl",
        "COG_category", "Description", "Preferred_name", "GOs", "EC", "KEGG_ko",
        "KEGG_Pathway", "KEGG_Module", "KEGG_Reaction", "KEGG_rclass", "BRITE",
        "KEGG_TC", "CAZy", "BiGG_Reaction", "PFAMs"]
lines = ["## hand-built fixture in eggNOG-mapper 2.1 layout", "#" + "\t".join(cols)]
for i, (qname, q) in enumerate(sorted(by_q.items())):
    names = q["names"] + (["NotAPfamName"] if i == 0 else [])
    vals = [qname, "9606.fixture", q["evalue"], f"{q['score']:.1f}"] + ["-"] * 16
    vals.append(",".join(names))
    lines.append("\t".join(vals))
(out / "eggnog_mapper.diamond.emapper.annotations").write_text("\n".join(lines) + "\n")
PYEOF

"$PY" "$HERE/bin/refdb_calls.py" pfam-names --pfam-hmm "$PFAM" --out "$TMP/pfam_names.tsv"
# Only the names the fixture uses, so the committed file stays small.
"$PY" - "$TMP/pfam_names.tsv" "$OUT" <<'PYEOF'
import sys
from pathlib import Path
names = dict(l.split("\t") for l in Path(sys.argv[1]).read_text().splitlines())
out = Path(sys.argv[2])
used = set()
for line in (out / "eggnog_mapper.diamond.emapper.annotations").read_text().splitlines():
    if not line.startswith("#"):
        used.update(line.split("\t")[-1].split(","))
keep = sorted(n for n in used if n in names)
(out / "pfam_names.tsv").write_text("".join(f"{n}\t{names[n]}\n" for n in keep))
PYEOF

"$PY" "$HERE/bin/build_domain_truth.py" --annotations "$FIX/annotations" \
  --truth-out "$TMP/human_domain_truth.parquet" --map-outdir "$TMP" \
  --summary-out "$TMP/truth_summary.json" > /dev/null

"$PY" "$HERE/bin/refdb_calls.py" interproscan --tsv "$OUT/interproscan.all_members.tsv.gz" \
  --queries "$FIX/qfo/human.fasta" --setting all_members \
  --out "$OUT/interproscan.all_members.regions.tsv.gz" \
  --qc-out "$OUT/interproscan.all_members.calls_qc.json" 2> /dev/null
"$PY" "$HERE/bin/refdb_calls.py" eggnog \
  --annotations "$OUT/eggnog_mapper.diamond.emapper.annotations" \
  --pfam-names "$OUT/pfam_names.tsv" --queries "$FIX/qfo/human.fasta" --setting diamond \
  --out "$OUT/eggnog_mapper.diamond.regions.tsv.gz" \
  --qc-out "$OUT/eggnog_mapper.diamond.calls_qc.json" 2> /dev/null

for pair in interproscan:all_members eggnog_mapper:diamond; do
  tool=${pair%%:*}; setting=${pair#*:}
  stem="pfam.$tool.$setting.own_models"
  "$PY" "$HERE/bin/evaluate_domain_calls.py" --regions "$OUT/$tool.$setting.regions.tsv.gz" \
    --tool "$tool" --variant "$setting" --species own_models \
    --truth "$TMP/human_domain_truth.parquet" --direct-annotation \
    --calls-out "$OUT/$stem.calls.parquet" --metrics-out "$OUT/$stem.metrics.parquet" \
    --curve-out "$OUT/$stem.curve.parquet" > /dev/null 2>&1
done
ls -la "$OUT"
