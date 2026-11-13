#!/usr/bin/env python3
"""Equal-depth test (notebook 361): does SPLASH reach the same genes when every sample has the same depth?

Usage: plot_equal_depth.py <equal_depth_reached.tsv> <equal_depth_control.tsv> <paper_deseq_small.parquet> <out_stem>

Inputs come from equal_depth_summary.py (Sherlock) and reached_genes.py (the paper's DESeq2 table).
a, b. Share of the paper's genes reached by SPLASH in each expression bin (the paper's baseMean),
      for young 24 h and young 2 h, one line per run:
        all samples, full depth (the original run)       solid line, filled circles
        kept pairs, full depth                            solid line, open circles
        kept pairs, 2 M read pairs per sample             dashed line, open circles
        kept pairs, 1 M read pairs per sample             dotted line, open circles
c.    Simulated control: share of genes in each planted class that SPLASH finds, against read pairs
      per sample.
Writes <out_stem>_bins.tsv with the numbers behind a and b, and prints them.
"""
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "notebooks"))
import pubfig as pf  # noqa: E402

reached_tsv, control_tsv, paper_pq, out_stem = sys.argv[1:]
CONTRASTS = ["young_24h", "young_2h"]
NAME = {"young_24h": "young 24 h", "young_2h": "young 2 h"}
BLUE = pf.OKABE_ITO["blue"]
ARMS = {  # arm: (label, line style, filled marker)
    "all_samples": ("all samples, full depth (original run)", "-", True),
    "full": ("kept pairs, full depth", "-", False),
    "2000000": ("kept pairs, 2 M read pairs per sample", "--", False),
    "1000000": ("kept pairs, 1 M read pairs per sample", ":", False),
}
BM_EDGES = [0, 1, 10, 100, 1000, np.inf]
BM_LABELS = ["<1", "1-10", "10-100", "100-1k", ">1k"]

reached = pl.read_csv(reached_tsv, separator="\t", schema_overrides={"arm": pl.Utf8})
paper = (pl.read_parquet(paper_pq).filter(pl.col("contrast").is_in(CONTRASTS) & (pl.col("baseMean") > 0))
         .with_columns(pl.col("baseMean").cut(BM_EDGES[1:-1], labels=BM_LABELS, left_closed=True)
                       .cast(pl.Utf8).alias("bin")))
rows = []
for arm in ARMS:
    hit = (reached.filter(pl.col("arm") == arm).select(pl.col("gene").alias("Gene_ID"), "contrast")
           .with_columns(pl.lit(True).alias("hit")))
    t = (paper.join(hit, on=["Gene_ID", "contrast"], how="left").with_columns(pl.col("hit").fill_null(False))
         .group_by("contrast", "bin").agg(pl.len().alias("genes"), pl.col("hit").sum().alias("reached"))
         .with_columns(pl.lit(arm).alias("arm"), (pl.col("reached") / pl.col("genes")).round(4).alias("share")))
    rows.append(t)
bins = pl.concat(rows).sort("contrast", "arm", "bin")
bins.write_csv(f"{out_stem}_bins.tsv", separator="\t")
with pl.Config(tbl_rows=100):
    print(bins)

pf.use_style()
fig, axs = pf.figure(pf.TWO_COLUMN_MM, 62, ncols=3, layout="constrained")
for j, c in enumerate(CONTRASTS):
    ax = axs[j]
    for arm, (lab, ls, filled) in ARMS.items():
        t = bins.filter((pl.col("contrast") == c) & (pl.col("arm") == arm))
        if t.height == 0:
            continue
        share = [t.filter(pl.col("bin") == b)["share"].item() if t.filter(pl.col("bin") == b).height else np.nan
                 for b in BM_LABELS]
        ax.plot(range(len(BM_LABELS)), share, linestyle=ls, marker="o", markersize=3, color=BLUE,
                markerfacecolor=BLUE if filled else "white", markeredgewidth=0.7, linewidth=0.9, label=lab)
    ax.set_xticks(range(len(BM_LABELS)), BM_LABELS)
    ax.set_ylim(0, 1)
    ax.set_xlabel("paper's baseMean (mean normalised count)")
    ax.set_title(NAME[c])
axs[0].set_ylabel("share of the paper's genes\nreached by SPLASH")
axs[1].sharey(axs[0])

ctrl = pl.read_csv(control_tsv, separator="\t", schema_overrides={"pairs_per_sample": pl.Utf8})
ctrl = ctrl.with_columns(pl.when(pl.col("pairs_per_sample").str.starts_with("full")).then(pl.lit("1000000"))
                         .otherwise(pl.col("pairs_per_sample")).cast(pl.Int64).alias("pairs"))
ax = axs[2]
t = ctrl.filter(pl.col("truth_class") == "isoform_switch").sort("pairs")
ax.plot(t["pairs"] / 1e6, t["recall"], marker="o", markersize=3, color=BLUE, linewidth=0.9)
ax.annotate("isoform switch", (t["pairs"][-1] / 1e6, t["recall"][-1]), xytext=(4, 0), textcoords="offset points",
            va="center", fontsize=5, color=BLUE)
# the other three classes are 0 at every depth; one line and one label say so
zero = ctrl.filter(pl.col("truth_class") != "isoform_switch")
assert (zero["recall"] == 0).all(), zero
ax.plot(t["pairs"] / 1e6, [0] * t.height, marker="s", markersize=3, color=BLUE, linewidth=0.9,
        markerfacecolor="white", markeredgewidth=0.7)
ax.annotate("higher in PEC, lower in PEC,\nand unchanged (false calls):\n0 at every depth", (t["pairs"][-1] / 1e6, 0),
            xytext=(4, 14), textcoords="offset points", va="bottom", fontsize=5, color=BLUE)
ax.set_xscale("log")
ax.set_xticks([0.125, 0.25, 0.5, 1], ["0.125", "0.25", "0.5", "1\n(full)"])
ax.minorticks_off()
ax.set_xlim(0.1, 4)
ax.set_ylim(0, 1)
ax.set_xlabel("read pairs per sample (millions)")
ax.set_ylabel("share of genes found by SPLASH")
ax.set_title("simulated control, 12 samples")
for a, l in zip(axs, "abc"):
    pf.panel_label(a, l)
fig.legend(*axs[0].get_legend_handles_labels(), loc="outside upper center", ncols=2, frameon=False)
pf.save(fig, out_stem)
print("wrote", out_stem)
