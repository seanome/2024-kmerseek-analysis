#!/usr/bin/env python3
"""Figure: share of each paper gene list reached, against the baseMean-matched null.

Usage: plot_paper_recovery.py <paper_recovery.tsv> <out_stem>

A 2 x 2 grid. Columns are the paper's up lists (genes higher in PEC) and its down lists (genes
lower in PEC). Rows are the two routes: SPLASH (blue) and anchor counts tested with DESeq2
(vermillion); the colours mean the same routes in the other figures. Within a panel, one row per
contrast. The triangle is the share of the real list reached, pointing up for an up list and down
for a down list. The grey bar is the middle 95 % of 1,000 random gene sets matched to the list's
baseMean deciles, and the black tick is their median. The text right of each row is
reached/listed and, in brackets, the share of random sets at or above the real list.
"""
import sys
from pathlib import Path

import polars as pl
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "notebooks"))
import pubfig as pf  # noqa: E402

tsv, out_stem = sys.argv[1:]
d = pl.read_csv(tsv, separator="\t")
CONTRASTS = ["young_2h", "young_24h", "old_24h", "young_13_49d", "young_132d", "old_41_93d"]
NAME = {"young_2h": "young 2 h", "young_24h": "young 24 h", "old_24h": "old 24 h",
        "young_13_49d": "young 13-49 d", "young_132d": "young 132 d", "old_41_93d": "old 41-93 d"}
ROUTE = {"splash": "SPLASH anchors (corrected p ≤ 0.05)",
         "anchor_nominal": "anchor counts, DESeq2 (p ≤ 0.05, 2-fold)"}
COLOUR = {"splash": pf.OKABE_ITO["blue"], "anchor_nominal": pf.OKABE_ITO["vermillion"]}
DIRECTION = {"up": ("▲ UP lists (higher in PEC)", "^"),
             "down": ("▼ DOWN lists (lower in PEC)", "v")}
routes = [c for c in ROUTE if c in set(d["channel"])]

pf.use_style()
fig = pf.figure(pf.TWO_COLUMN_MM, 30 + 32 * len(routes))[0]
fig.clear()

# Legend first, so it sits above every mark in reading order.
handles = [
    Line2D([], [], linestyle="none", marker="^", color="black", markersize=5,
           label="real up list"),
    Line2D([], [], linestyle="none", marker="v", color="black", markersize=5,
           label="real down list"),
    Line2D([], [], color=pf.GREY, linewidth=3, solid_capstyle="butt",
           label="middle 95 % of 1,000 baseMean-matched random sets"),
    Line2D([], [], linestyle="none", marker="|", color="black", markersize=6,
           label="median of random sets"),
] + [Line2D([], [], linestyle="none", marker="s", color=COLOUR[r], markersize=5,
            label="SPLASH" if r == "splash" else "anchor counts, DESeq2") for r in routes]
fig.legend(handles=handles, loc="outside upper center", ncol=3,
           title="text right of each row: reached/listed (share of random sets at or above the real list)")

subfigs = fig.subfigures(1, 2, wspace=0.02)
for sf, direction in zip(subfigs, ("up", "down")):
    header, marker = DIRECTION[direction]
    sf.suptitle(header, fontweight="bold", fontsize=8)
    axs = sf.subplots(len(routes), 1, sharex=True, squeeze=False)[:, 0]
    for ax, route in zip(axs, routes):
        for y, c in enumerate(CONTRASTS):
            r = d.filter((pl.col("channel") == route) & (pl.col("contrast") == c)
                         & (pl.col("list") == direction))
            if r.height == 0:
                continue
            r = r.row(0, named=True)
            ax.plot([r["null_2_5"], r["null_97_5"]], [y, y], color=pf.GREY, linewidth=3,
                    solid_capstyle="butt")
            ax.plot([r["null_median"]], [y], marker="|", color="black", markersize=6, clip_on=False)
            ax.plot([r["fraction_reached"]], [y], marker=marker, color=COLOUR[route],
                    markersize=5, linestyle="none", clip_on=False, zorder=3)
            ax.text(1.02, y,
                    f"{r['reached']}/{r['list_genes_with_protein']}  "
                    f"({r['share_null_at_or_above']:.3f})",
                    transform=ax.get_yaxis_transform(), va="center")
        ax.set_title(ROUTE[route], loc="left")
        ax.set_xlim(0, 1)
        ax.set_ylim(len(CONTRASTS) - 0.5, -0.5)
        ax.set_yticks(range(len(CONTRASTS)), [NAME[c] for c in CONTRASTS])
        ax.tick_params(axis="y", pad=5)  # room for a triangle sitting on 0
    axs[-1].set_xlabel("share of the paper's list reached")
pf.save(fig, out_stem)
print("wrote", out_stem)
