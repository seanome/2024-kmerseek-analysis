#!/usr/bin/env python3
"""Every old gene model reached by each route's supported peptides, per contrast.

Usage: reached_genes.py <contrasts_dir> <botznik.id_map.tsv> <paper_deseq_all.parquet> <out_dir>

A gene is reached when one of the route's peptides aligns to one of its old gene-model
proteins with `supported` true (E <= 1e-5, query coverage >= 50 %), the same rule as
paper_recovery. Writes <out_dir>/reached_genes.tsv (contrast, route, gene, peptides) and
<out_dir>/paper_deseq_small.parquet (the paper's DESeq2 columns needed to plot against).
"""
import sys
from pathlib import Path

import polars as pl

contrasts_dir, id_map, paper, out = Path(sys.argv[1]), sys.argv[2], sys.argv[3], Path(sys.argv[4])
out.mkdir(parents=True, exist_ok=True)
CONTRASTS = ["young_2h", "young_24h", "old_24h", "young_13_49d", "young_132d", "old_41_93d"]
ROUTES = {"splash": "search_compactors", "anchor_nominal": "search_compactors_anchor_nominal",
          "dekupl_limma": "search_dekupl_limma", "kamrat_ttest": "search_kamrat_ttest"}
genes = (pl.read_csv(id_map, separator="\t", columns=["stable_id", "original_id"])
         .with_columns(pl.col("original_id").str.replace(r"\.t\d+$", "").alias("gene")))
parts = []
for c in CONTRASTS:
    for route, sub in ROUTES.items():
        f = contrasts_dir / c / sub / "align.botznik.parquet"
        if not f.exists():
            print(c, route, "no alignment file")
            continue
        r = (pl.scan_parquet(f).filter(pl.col("supported")).select("query", "target")
             .join(genes.lazy(), left_on="target", right_on="stable_id")
             .group_by("gene").agg(pl.col("query").n_unique().alias("peptides"))
             .with_columns(pl.lit(c).alias("contrast"), pl.lit(route).alias("route")).collect())
        print(c, route, r.height, "genes")
        parts.append(r.select("contrast", "route", "gene", "peptides"))
pl.concat(parts).write_csv(out / "reached_genes.tsv", separator="\t")
(pl.read_parquet(paper, columns=["Gene_ID", "Gene_Name", "contrast", "baseMean", "log2FoldChange", "pvalue", "padj"])
 .write_parquet(out / "paper_deseq_small.parquet"))
print("wrote", out)
