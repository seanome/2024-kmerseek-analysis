#!/usr/bin/env python3
"""Genes reached with the real PEC/control labels and with swapped labels (notebook 361).

Usage: label_swap_reach.py <pec_root> <manifest.tsv> <botznik.id_map.tsv> <paper_deseq_all.parquet> <out_dir>

For each contrast in the manifest, reads the real run (<pec_root>/contrasts/<contrast>) and each
swap (<pec_root>/label_swap/<contrast>/<flip>), for two routes:
  splash          every anchor SPLASH keeps (corrected p <= 0.05) -> compactors -> peptides
  anchor_nominal  DESeq2 on anchor counts, p <= 0.05 and 2-fold -> compactors -> peptides
A gene is reached when one of the route's peptides aligns to one of its old gene-model proteins
with `supported` true (E <= 1e-5, query coverage >= 50 %), the rule of reached_genes.py.
A run whose search has not finished is written with empty counts.

Writes to <out_dir>:
  label_swap_runs.tsv  one row per contrast, run and route: kept anchors, compactors, peptides,
                       genes reached, genes with the paper's baseMean > 1000 and the share reached
  label_swap_bins.tsv  genes and genes reached per paper baseMean bin (row c of reached_vs_paper_de)
"""
import sys
from pathlib import Path

import numpy as np
import polars as pl

root, manifest, id_map, paper_pq, out = (Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4], Path(sys.argv[5]))
out.mkdir(parents=True, exist_ok=True)
BM_EDGES = [1, 10, 100, 1000]
BM_LABELS = ["<1", "1-10", "10-100", "100-1k", ">1k"]
ROUTES = {"splash": ("compactors", "search_compactors"),
          "anchor_nominal": ("compactors_anchor_nominal", "search_compactors_anchor_nominal")}

man = pl.read_csv(manifest, separator="\t")
genes = (pl.read_csv(id_map, separator="\t", columns=["stable_id", "original_id"])
         .with_columns(pl.col("original_id").str.replace(r"\.t\d+$", "").alias("gene")))
paper = (pl.read_parquet(paper_pq, columns=["Gene_ID", "contrast", "baseMean"])
         .filter(pl.col("baseMean") > 0)
         .with_columns(pl.col("baseMean").cut(BM_EDGES, labels=BM_LABELS, left_closed=True)
                       .cast(pl.Utf8).alias("bin")))


def data_rows(path):
    """Rows after the header line, or None if the file is missing."""
    if not path.exists():
        return None
    with open(path, "rb") as fh:
        return sum(1 for _ in fh) - 1


def n_fasta(path):
    if not path.exists():
        return None
    with open(path, "rb") as fh:
        return sum(1 for line in fh if line.startswith(b">"))


runs = [(c, "real", root / "contrasts" / c) for c in man["contrast"].unique(maintain_order=True)]
runs += [(r["contrast"], r["flip"], root / "label_swap" / r["contrast"] / r["flip"]) for r in man.iter_rows(named=True)]
ratio = {(r["contrast"], r["flip"]): r["control_over_PEC_reads"] for r in man.iter_rows(named=True)}
ratio.update({(r["contrast"], "real"): r["real_control_over_PEC_reads"] for r in man.iter_rows(named=True)})

rows, bins = [], []
for c, run, d in runs:
    pc = paper.filter(pl.col("contrast") == c)
    for route, (cdir, sdir) in ROUTES.items():
        kept = (data_rows(d / "splash_out" / "result.after_correction.scores.tsv") if route == "splash"
                else data_rows(d / cdir / "anchors_nominal.tsv"))
        row = dict(contrast=c, run=run, route=route, control_over_PEC_reads=ratio[(c, run)], kept_anchors=kept,
                   compactors=data_rows(d / cdir / "compactors.tsv"), peptides=None, genes_reached=None,
                   genes_basemean_gt_1000=None, reached_basemean_gt_1000=None, share_reached_basemean_gt_1000=None)
        done = (d / sdir / "search.done").exists()
        if done:
            row["peptides"] = n_fasta(d / sdir / "pep.peptides.faa") or 0
            f = d / sdir / "align.botznik.parquet"
            if f.exists():
                hit = (pl.scan_parquet(f).filter(pl.col("supported")).select("target")
                       .join(genes.lazy(), left_on="target", right_on="stable_id")
                       .select("gene").unique().collect())
            else:  # no peptides, so nothing searched
                hit = pl.DataFrame({"gene": []}, schema={"gene": pl.Utf8})
            row["genes_reached"] = hit.height
            g = pc.with_columns(pl.col("Gene_ID").is_in(hit["gene"].implode()).alias("reached"))
            hi = g.filter(pl.col("baseMean") > 1000)
            row["genes_basemean_gt_1000"] = hi.height
            row["reached_basemean_gt_1000"] = int(hi["reached"].sum())
            row["share_reached_basemean_gt_1000"] = round(hi["reached"].mean(), 4) if hi.height else np.nan
            bins.append(g.group_by("bin").agg(pl.len().alias("genes"), pl.col("reached").sum().alias("reached"))
                        .with_columns(pl.lit(c).alias("contrast"), pl.lit(run).alias("run"), pl.lit(route).alias("route")))
        print(row)
        rows.append(row)
pl.DataFrame(rows).write_csv(out / "label_swap_runs.tsv", separator="\t")
if bins:
    (pl.concat(bins).select("contrast", "run", "route", "bin", "genes", "reached")
     .sort("contrast", "run", "route", "bin").write_csv(out / "label_swap_bins.tsv", separator="\t"))
print("wrote", out)
