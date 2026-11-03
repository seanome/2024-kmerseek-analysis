#!/usr/bin/env python3
"""Figures for the human contrasts of notebook 361.

Usage: plot_human.py <human_genes_summary.tsv> <human_compare.tsv> <contrasts.tsv> <out_dir>

human_routes: per contrast (rows, grouped by series) and route (columns), a: sequences kept,
b: genes reached (a supported peptide match to a GENCODE v50 protein). Up triangle: higher in the
study's second group; down triangle: lower; circle: SPLASH, which has no direction. Colour is
the tool: SPLASH blue, KaMRaT reddish purple (the same colours as the Botryllus figures).
human_recovery: share of each study's own up list (up triangles) and down list (down
triangles) reached by each route; grey bar the middle 95 % of 1,000 random gene sets matched to
the list's baseMean deciles, black tick their median.
"""
import sys
from pathlib import Path

import numpy as np
import polars as pl
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "notebooks"))
import pubfig as pf  # noqa: E402

summ_p, comp_p, con_p, out = sys.argv[1], sys.argv[2], sys.argv[3], Path(sys.argv[4])
out.mkdir(parents=True, exist_ok=True)
summ = pl.read_csv(summ_p, separator="\t")
comp = pl.read_csv(comp_p, separator="\t") if Path(comp_p).exists() else None
con = pl.read_csv(con_p, separator="\t")
ROUTES = {"splash_compactors": "SPLASH anchors\ncorrected p ≤ 0.05",
          "kamrat_ttest_corrected": "KaMRaT, t-test\ncorrected p ≤ 0.05",
          "kamrat_ttest_p05": "KaMRaT, t-test\np < 0.05",
          "kamrat_deseq2_corrected": "KaMRaT, DESeq2\ncorrected p ≤ 0.05",
          "kamrat_deseq2_p05": "KaMRaT, DESeq2\np < 0.05"}
COL = {r: (pf.OKABE_ITO["blue"] if r.startswith("splash") else pf.OKABE_ITO["reddish_purple"]) for r in ROUTES}
MARK = {"up": "^", "down": "v", "none": "o"}
contrasts = [c for c in con["contrast"] if c in set(summ["contrast"])]
label = {r["contrast"]: f"{r['series']}  {r['other']} vs {r['reference']}" + (
    "" if r["series"] == "GSE120862" else "") for r in con.iter_rows(named=True)}
for r in con.iter_rows(named=True):
    c = r["contrast"]
    extra = c.split("_", 1)[1].rsplit("_", 3)[0] if r["series"] != "GSE99963" else c.split("_")[1]
    label[c] = f"{r['series']} {extra.replace('_', ' ')}: {r['other']} vs {r['reference']}"
routes = [r for r in ROUTES if r in set(summ["route"])]

pf.use_style()
y = {c: i for i, c in enumerate(contrasts)}
h = min(pf.MAX_HEIGHT_MM, 28 + 2 * 5.5 * len(contrasts))
fig, axes = pf.figure(pf.TWO_COLUMN_MM, h, nrows=2, ncols=len(routes), sharey=True, squeeze=False)
handles = [Line2D([], [], linestyle="none", marker="^", color="black", markersize=4, label="higher in second group"),
           Line2D([], [], linestyle="none", marker="v", color="black", markersize=4, label="lower in second group"),
           Line2D([], [], linestyle="none", marker="o", color="black", markersize=4, label="SPLASH (no direction)"),
           Line2D([], [], linestyle="none", marker="s", color=pf.OKABE_ITO["blue"], markersize=4, label="SPLASH"),
           Line2D([], [], linestyle="none", marker="s", color=pf.OKABE_ITO["reddish_purple"], markersize=4, label="KaMRaT")]
fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.0))
for row, (col, xlab) in enumerate((("sequences_kept", "sequences kept (+1)"), ("genes", "genes reached (+1)"))):
    for j, r in enumerate(routes):
        ax = axes[row][j]
        d = summ.filter(pl.col("route") == r)
        for t in d.iter_rows(named=True):
            if t["contrast"] not in y:
                continue
            ax.scatter(t[col] + 1, y[t["contrast"]], marker=MARK[t["direction"]], color=COL[r], s=12, zorder=3)
        ax.set_xscale("log")
        ax.set_xlabel(xlab)
        ax.grid(axis="x", color="#dddddd", linewidth=0.4)
        if row == 0:
            ax.set_title(ROUTES[r])
        ax.set_yticks(range(len(contrasts)))
        ax.set_yticklabels([label[c] for c in contrasts])
        ax.invert_yaxis() if j == 0 else None
    pf.panel_label(axes[row][0], "ab"[row], dx_pt=-150)
fig.subplots_adjust(top=0.88, left=0.27, right=0.99, hspace=0.45, wspace=0.12)
pf.save(fig, str(out / "human_routes"))

if comp is not None and comp.height:
    cr = [r for r in ROUTES if r in set(comp["route"])]
    fig, axes = pf.figure(pf.TWO_COLUMN_MM, min(pf.MAX_HEIGHT_MM, 30 + 5.5 * len(contrasts)), ncols=len(cr),
                          sharey=True, squeeze=False)
    handles = [Line2D([], [], linestyle="none", marker="^", color="black", markersize=4, label="study's up list"),
               Line2D([], [], linestyle="none", marker="v", color="black", markersize=4, label="study's down list"),
               Line2D([], [], color=pf.GREY, linewidth=3, solid_capstyle="butt", label="middle 95 % of random gene sets"),
               Line2D([], [], linestyle="none", marker="|", color="black", markersize=5, label="median of random sets")]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.0))
    for j, r in enumerate(cr):
        ax = axes[0][j]
        d = comp.filter(pl.col("route") == r)
        for t in d.iter_rows(named=True):
            if t["contrast"] not in y or not t["list_genes"]:
                continue
            yy = y[t["contrast"]] + (-0.18 if t["direction"] == "up" else 0.18)
            ax.plot([t["null_lo"], t["null_hi"]], [yy, yy], color=pf.GREY, linewidth=2.5, solid_capstyle="butt", zorder=1)
            ax.plot([t["null_median"]], [yy], marker="|", color="black", markersize=4, zorder=2)
            ax.plot([t["share"]], [yy], marker=MARK[t["direction"]], color=COL[r], markersize=4, linestyle="none", zorder=3)
        ax.set_title(ROUTES[r])
        ax.set_xlabel("share of the study's list reached")
        ax.set_yticks(range(len(contrasts)))
        ax.set_yticklabels([label[c] for c in contrasts])
        ax.set_xlim(left=0)
        ax.grid(axis="x", color="#dddddd", linewidth=0.4)
    axes[0][0].set_ylim(len(contrasts) - 0.5, -0.5)
    fig.subplots_adjust(top=0.86, left=0.27, right=0.99, wspace=0.12)
    pf.save(fig, str(out / "human_recovery"))
print("wrote", out)
