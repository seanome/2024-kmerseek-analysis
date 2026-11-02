#!/usr/bin/env python3
"""Per contrast: KaMRaT contigs at uncorrected p, peptides, genes reached, and paper-list genes reached.

Usage: kamrat_nominal_table.py <contrasts_dir> <reached_genes.tsv> <paper_named_lists.tsv> <out.tsv>

Reads <contrasts_dir>/<c>/kamrat_nominal/summary.json (select_kamrat_nominal.py) and counts the
peptides in <c>/search_kamrat_nominal/pep.peptides.faa. Genes reached come from reached_genes.tsv
(reached_genes.py). For comparison the same gene counts are given for SPLASH and anchor counts.
A paper-list gene counts as reached when the route reaches it at all, in either direction.
"""
import json
import sys
from pathlib import Path

import polars as pl

contrasts_dir, reached_tsv, lists_tsv, out = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
CONTRASTS = ["young_2h", "young_24h", "old_24h", "young_13_49d", "young_132d", "old_41_93d"]
ROUTES = ["kamrat_nominal", "anchor_nominal", "splash"]
reached = pl.read_csv(reached_tsv, separator="\t")
lists = pl.read_csv(lists_tsv, separator="\t")
rows = []
for c in CONTRASTS:
    s = json.loads((contrasts_dir / c / "kamrat_nominal" / "summary.json").read_text())
    faa = contrasts_dir / c / "search_kamrat_nominal" / "pep.peptides.faa"
    pep = sum(1 for line in open(faa) if line.startswith(">")) if faa.exists() else 0
    row = {"contrast": c, "kamrat_contigs_tested": s["contigs_tested"], "kamrat_p_le_0.05": s["p_le_0.05"],
           "kamrat_p_le_0.05_2fold": s["p_le_0.05_and_2fold"], "kamrat_up": s["up"], "kamrat_down": s["down"],
           "kamrat_peptides": pep}
    for d in ["up", "down"]:
        row[f"paper_{d}_listed"] = lists.filter((pl.col("contrast") == c) & (pl.col("direction") == d))["gene_id"].n_unique()
    for r in ROUTES:
        g = set(reached.filter((pl.col("contrast") == c) & (pl.col("route") == r))["gene"])
        row[f"{r}_genes"] = len(g)
        for d in ["up", "down"]:
            listed = set(lists.filter((pl.col("contrast") == c) & (pl.col("direction") == d))["gene_id"])
            row[f"{r}_paper_{d}_reached"] = len(listed & g)
    rows.append(row)
t = pl.DataFrame(rows)
t.write_csv(out, separator="\t")
with pl.Config(tbl_cols=-1, tbl_width_chars=250):
    print(t.transpose(include_header=True, column_names="contrast"))
