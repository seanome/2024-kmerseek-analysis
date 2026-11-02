#!/usr/bin/env python3
"""Which of the paper's genes do the k-mer routes reach: the strongly changed ones, or the highly expressed ones?

Usage: plot_reached_vs_paper_de.py <reached_genes.tsv> <paper_deseq_small.parquet> <out_stem>

Inputs come from reached_genes.py. DE-kupl and KaMRaT keep 0 sequences in every real contrast
after correction, so the routes shown are SPLASH, anchor counts (DESeq2 at p <= 0.05, 2-fold) and
KaMRaT at the same uncorrected cutoff (its t-test p <= 0.05, 2-fold; select_kamrat_nominal.py).

Rows of panels, one column per contrast:
a-c. The paper's DESeq2 log2 fold change (PEC over control) against its mean normalised count
     (baseMean), every gene with baseMean > 0 in light grey, genes reached by the route on top.
d.   Share of genes reached in each expression bin.
e.   Among genes with baseMean >= 10, share reached in bins of the paper's p-value.
Writes <out_stem>_bins.tsv with the numbers behind c and d, and prints them.
"""
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "notebooks"))
import pubfig as pf  # noqa: E402

reached_tsv, paper_pq, out_stem = sys.argv[1:]
CONTRASTS = ["young_2h", "young_24h", "old_24h", "young_13_49d", "young_132d", "old_41_93d"]
NAME = {"young_2h": "young 2 h", "young_24h": "young 24 h", "old_24h": "old 24 h",
        "young_13_49d": "young 13-49 d", "young_132d": "young 132 d", "old_41_93d": "old 41-93 d"}
ROUTE = {"splash": "SPLASH", "anchor_nominal": "anchor counts, DESeq2",
         "kamrat_nominal": "KaMRaT, uncorrected p"}
COLOUR = {"splash": pf.OKABE_ITO["blue"], "anchor_nominal": pf.OKABE_ITO["vermillion"],
          "kamrat_nominal": pf.OKABE_ITO["reddish_purple"]}
N_MA = len(ROUTE)  # one MA-plot row per route, then the two share rows
BM_EDGES = [0, 1, 10, 100, 1000, np.inf]
BM_LABELS = ["<1", "1-10", "10-100", "100-1k", ">1k"]
P_EDGES = [0, 0.001, 0.01, 0.05, 0.2, 1.000001]
P_LABELS = ["<0.001", "0.001-0.01", "0.01-0.05", "0.05-0.2", ">0.2"]

reached = pl.read_csv(reached_tsv, separator="\t")
paper = pl.read_parquet(paper_pq).filter(pl.col("baseMean") > 0)
for route in ROUTE:
    hit = reached.filter(pl.col("route") == route).select(pl.col("gene").alias("Gene_ID"), "contrast",
                                                          pl.lit(True).alias(route))
    paper = paper.join(hit, on=["Gene_ID", "contrast"], how="left").with_columns(pl.col(route).fill_null(False))
paper = paper.with_columns(
    pl.col("baseMean").cut(BM_EDGES[1:-1], labels=BM_LABELS, left_closed=True).alias("expression_bin"),
    pl.col("pvalue").cut(P_EDGES[1:-1], labels=P_LABELS, left_closed=True).alias("pvalue_bin"))

bins = []
for kind, col, labels, sub in [("expression", "expression_bin", BM_LABELS, paper),
                               ("pvalue_baseMean_ge_10", "pvalue_bin", P_LABELS,
                                paper.filter((pl.col("baseMean") >= 10) & pl.col("pvalue").is_not_null()))]:
    t = (sub.group_by("contrast", col).agg(pl.len().alias("genes"), *[pl.col(r).sum().alias(f"{r}_reached") for r in ROUTE])
         .rename({col: "bin"}).with_columns(pl.col("bin").cast(pl.Utf8), pl.lit(kind).alias("binning")))
    bins.append(t)
bins = pl.concat(bins).sort("binning", "contrast", "bin")
bins.write_csv(f"{out_stem}_bins.tsv", separator="\t")
with pl.Config(tbl_rows=100):
    print(bins)

pf.use_style()
fig, axs = pf.figure(pf.TWO_COLUMN_MM, 170, nrows=N_MA + 2, ncols=len(CONTRASTS), sharex="row", sharey="row")
for j, c in enumerate(CONTRASTS):
    x = paper.filter((pl.col("contrast") == c) & pl.col("log2FoldChange").is_not_null())
    for i, route in enumerate(ROUTE):
        ax = axs[i, j]
        ax.scatter(x["baseMean"], x["log2FoldChange"], s=0.3, color="#dddddd", rasterized=True, linewidths=0)
        h = x.filter(pl.col(route))
        ax.scatter(h["baseMean"], h["log2FoldChange"], s=0.6, color=COLOUR[route], rasterized=True, linewidths=0)
        ax.axhline(0, color="black", linewidth=0.4)
        ax.set_xscale("log")
        ax.set_ylim(-12, 12)
        if i == 0:
            ax.set_title(NAME[c])
        ax.text(0.03, 0.97, f"{h.height:,} reached", transform=ax.transAxes, va="top", fontsize=5)
    for i, (kind, labels) in enumerate([("expression", BM_LABELS), ("pvalue_baseMean_ge_10", P_LABELS)], start=N_MA):
        ax = axs[i, j]
        t = bins.filter((pl.col("binning") == kind) & (pl.col("contrast") == c))
        for route in ROUTE:
            share = [(lambda r: r[f"{route}_reached"] / r["genes"] if r else np.nan)(
                t.filter(pl.col("bin") == b).row(0, named=True) if t.filter(pl.col("bin") == b).height else None)
                for b in labels]
            ax.plot(range(len(labels)), share, marker="o", markersize=2, color=COLOUR[route])
        ax.set_xticks(range(len(labels)), labels, rotation=60, ha="right")
        ax.set_ylim(0, 1)
for i, lab in enumerate(["log2 fold change\n(paper, PEC / ctrl)"] * N_MA
                        + ["share of genes\nreached", "share reached\n(baseMean ≥ 10)"]):
    axs[i, 0].set_ylabel(lab)
for j in range(len(CONTRASTS)):
    for i in range(N_MA - 1):
        axs[i, j].tick_params(labelbottom=False)
    axs[N_MA - 1, j].set_xlabel("paper's baseMean")
    axs[N_MA, j].set_xlabel("paper's baseMean")
    axs[N_MA + 1, j].set_xlabel("paper's p-value")
for i, l in enumerate("abcde"):
    pf.panel_label(axs[i, 0], l)
handles = [pf.plt.Line2D([], [], marker="o", linestyle="none", color="#dddddd", label="gene not reached"),
           *[pf.plt.Line2D([], [], marker="o", color=COLOUR[r], label=f"reached by {ROUTE[r]}") for r in ROUTE]]
fig.legend(handles=handles, loc="outside upper center", ncols=4, frameon=False)
pf.save(fig, out_stem)
print("wrote", out_stem)
