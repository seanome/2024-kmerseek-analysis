#!/usr/bin/env python3
"""Share of genes reached by expression bin, real PEC/control labels against swapped labels.

Usage: plot_label_swap_reach.py <label_swap_bins.tsv> <label_swap_runs.tsv> <out_stem>

Inputs from label_swap_reach.py. One column per contrast; row a SPLASH, row b anchor counts
(DESeq2, p <= 0.05 and 2-fold). Real labels: solid line, filled circles. Each swap: dashed line,
open markers, one marker shape per swap (picked by control/PEC read ratio, see
design/label_swap/manifest.tsv). Same layout as row c of reached_vs_paper_de.

Row b marks with "+" each swap whose DESeq2 design (~ pair + condition; young 132 d ~ genotype +
condition) has no flipped colony pair (or genotype) among those holding both labels: its test then
rests only on pairs with their real labels, so it is not a null. Read from design/ (added 2026-11-03).
"""
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "notebooks"))
import pubfig as pf  # noqa: E402

bins_tsv, runs_tsv, out_stem = sys.argv[1:]
CONTRASTS = ["young_2h", "young_24h", "old_24h", "young_13_49d", "young_132d", "old_41_93d"]
NAME = {"young_2h": "young 2 h", "young_24h": "young 24 h", "old_24h": "old 24 h",
        "young_13_49d": "young 13-49 d", "young_132d": "young 132 d", "old_41_93d": "old 41-93 d"}
ROUTE = {"splash": "SPLASH", "anchor_nominal": "anchor counts, DESeq2"}
COLOUR = {"splash": pf.OKABE_ITO["blue"], "anchor_nominal": pf.OKABE_ITO["vermillion"]}
SWAP = {"swap_uneven": ("s", "swap with the most uneven read split"),
        "swap_middle": ("^", "swap with a middle read split"),
        "swap_even": ("D", "swap with the most even read split")}
BM_LABELS = ["<1", "1-10", "10-100", "100-1k", ">1k"]

DESIGN = Path(__file__).resolve().parents[1] / "design"


def real_label_only(contrast, flip):
    """True if every block (colony pair; genotype in young 132 d) holding both labels keeps its real labels."""
    blk = "genotype" if contrast == "young_132d" else "pair"
    real = (pl.read_csv(DESIGN / f"contrast_{contrast}.design.tsv", separator="\t")
            .select("sample", pl.col("condition").alias("real")))
    sw = pl.read_csv(DESIGN / "label_swap" / f"{contrast}.{flip}.design.tsv", separator="\t").join(real, on="sample")
    both = sw.group_by(blk).agg(pl.col("condition").n_unique().alias("n"),
                                (pl.col("condition") != pl.col("real")).any().alias("flipped"))
    return both.filter(pl.col("n") == 2)["flipped"].sum() == 0


bins = pl.read_csv(bins_tsv, separator="\t")
runs = pl.read_csv(runs_tsv, separator="\t")
contrasts = [c for c in CONTRASTS if c in set(bins["contrast"])]

pf.use_style()
fig, axs = pf.figure(pf.TWO_COLUMN_MM, 95, nrows=2, ncols=len(contrasts), sharex=True, sharey=True, squeeze=False)
for j, c in enumerate(contrasts):
    for i, route in enumerate(ROUTE):
        ax = axs[i, j]
        t = bins.filter((pl.col("contrast") == c) & (pl.col("route") == route))
        for run in ["real", *SWAP]:
            r = t.filter(pl.col("run") == run)
            if r.height == 0:
                continue
            share = [(lambda x: x["reached"][0] / x["genes"][0] if x.height else np.nan)(r.filter(pl.col("bin") == b))
                     for b in BM_LABELS]
            if run == "real":
                ax.plot(range(5), share, marker="o", markersize=3, color=COLOUR[route], zorder=3)
            else:
                ax.plot(range(5), share, marker=SWAP[run][0], markersize=3, color=COLOUR[route], linestyle="--",
                        markerfacecolor="white", linewidth=0.7, zorder=2)
        # control/PEC read ratio of each run, top-left
        rr = runs.filter((pl.col("contrast") == c) & (pl.col("route") == route) & pl.col("genes_reached").is_not_null())
        lines = []
        for run in ["real", *SWAP]:
            x = rr.filter(pl.col("run") == run)
            if x.height:
                v = x.row(0, named=True)
                name = "real" if run == "real" else {"swap_uneven": "uneven", "swap_middle": "middle", "swap_even": "even"}[run]
                mark = " +" if route == "anchor_nominal" and run != "real" and real_label_only(c, run) else ""
                lines.append(f"{name} {v['control_over_PEC_reads']:.2f}: "
                             f"{100 * v['share_reached_basemean_gt_1000']:.0f}%{mark}")
        ax.text(0.03, 0.97, "ctrl/PEC reads:\n>1k reached\n" + "\n".join(lines), transform=ax.transAxes,
                va="top", fontsize=4.5)
        ax.set_ylim(0, 1.04)
        ax.set_xticks(range(5), BM_LABELS, rotation=60, ha="right")
        if i == 0:
            ax.set_title(NAME[c])
        if i == 1:
            ax.set_xlabel("paper's baseMean")
axs[0, 0].set_ylabel("share of genes reached\nSPLASH")
axs[1, 0].set_ylabel("share of genes reached\nanchor counts, DESeq2")
pf.panel_label(axs[0, 0], "a")
pf.panel_label(axs[1, 0], "b")
handles = [pf.plt.Line2D([], [], marker="o", color="black", markersize=3, label="real labels (filled, solid)")]
handles += [pf.plt.Line2D([], [], marker=m, color="black", markersize=3, linestyle="--", markerfacecolor="white",
                          linewidth=0.7, label=f"{lab} (open, dashed)") for m, lab in SWAP.values()]
handles += [pf.plt.Line2D([], [], color=COLOUR[r], label=ROUTE[r]) for r in ROUTE]
fig.legend(handles=handles, loc="outside upper center", ncols=3, frameon=False)
# supxlabel, unlike fig.text, gets its own space from the constrained layout
fig.supxlabel("Text in each panel: control/PEC read ratio of the run, then the share of genes with the paper's"
              " baseMean above 1,000 that it reaches.  + : this swap's DESeq2 test uses only colony pairs that"
              " keep their real labels, so it is not a null.", fontsize=5)
pf.save(fig, out_stem)
print("wrote", out_stem)
