#!/usr/bin/env python3
"""Figures for the human contrasts of notebook 361.

Usage: plot_human.py <human_genes_summary.tsv> <human_compare.tsv> <contrasts.tsv> <out_dir>

human_routes: one row per contrast (grouped by series), one column per route.
  a: sequences each route kept; b: genes reached (a peptide of 20 or more residues matching a
  GENCODE v50 protein, E <= 1e-5 over >= 50 % of the peptide).
  Up triangle: higher in the study's second group; down triangle: lower; circle: SPLASH, which
  has no direction. Colour is the tool: SPLASH blue, KaMRaT reddish purple (as in the Botryllus
  figures). Counts are drawn on a log axis as count + 1, so 0 sits at 1.
human_recovery: share of each study's up list (up triangles) and down list (down triangles)
  that each route reaches; grey bar the middle 95 % of 1,000 random gene sets matched to the
  list's baseMean deciles, black tick their median.
"""
import sys
from pathlib import Path

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
SHORT = {"gse99963_sed_post_vs_pre": "high-fat diet: after vs before",
         "gse99963_ex_post_vs_pre": "diet + exercise: after vs before",
         "gse144304_female_frail_vs_fit": "old women: frail vs fit",
         "gse144304_male_frail_vs_fit": "old men: frail vs fit",
         "gse144304_female_old_vs_young": "women: old vs young",
         "gse144304_male_old_vs_young": "men: old vs young",
         "gse120862_untrained_1h_ex_vs_ne": "untrained 1 h: exercised vs rested leg",
         "gse120862_untrained_4h_ex_vs_ne": "untrained 4 h: exercised vs rested leg",
         "gse120862_trained_1h_ex_vs_ne": "trained 1 h: exercised vs rested leg",
         "gse120862_trained_4h_ex_vs_ne": "trained 4 h: exercised vs rested leg",
         "gse120862_baseline_trained_vs_untrained": "rest: after vs before training"}
series_of = dict(zip(con["contrast"], con["series"]))
contrasts = [c for c in con["contrast"] if c in set(summ["contrast"])]
labels = [f"{series_of[c]}  {SHORT[c]}" for c in contrasts]
y = {c: i for i, c in enumerate(contrasts)}
routes = list(ROUTES)
pf.use_style()

KEY = [Line2D([], [], linestyle="none", marker="^", color="black", markersize=4, label="higher in second group"),
       Line2D([], [], linestyle="none", marker="v", color="black", markersize=4, label="lower in second group"),
       Line2D([], [], linestyle="none", marker="o", color="black", markersize=4, label="no direction"),
       Line2D([], [], linestyle="none", marker="s", color=pf.OKABE_ITO["blue"], markersize=4, label="SPLASH"),
       Line2D([], [], linestyle="none", marker="s", color=pf.OKABE_ITO["reddish_purple"], markersize=4, label="KaMRaT")]

row_h = 3.2
h = min(pf.MAX_HEIGHT_MM, 30 + 2 * (12 + row_h * len(contrasts)))
fig, axes = pf.figure(pf.TWO_COLUMN_MM, h, nrows=2, ncols=len(routes), sharey=True, squeeze=False)
fig.legend(handles=KEY, loc="outside upper center", ncol=5, frameon=False)
for row, (col, xlab, xmax) in enumerate((("sequences_kept", "sequences kept + 1", 1e6),
                                          ("genes", "genes reached + 1", 1e4))):
    for j, r in enumerate(routes):
        ax = axes[row][j]
        for t in summ.filter(pl.col("route") == r).iter_rows(named=True):
            if t["contrast"] in y:
                ax.scatter(t[col] + 1, y[t["contrast"]], marker=MARK[t["direction"]], color=COL[r], s=10,
                           zorder=3, linewidths=0)
        ax.set_xscale("log")
        ax.set_xlim(0.7, xmax)
        ax.set_xlabel(xlab)
        ax.grid(axis="x", color="#dddddd", linewidth=0.4)
        if row == 0:
            ax.set_title(ROUTES[r])
    axes[row][0].set_yticks(range(len(contrasts)))
    axes[row][0].set_yticklabels(labels)
    axes[row][0].set_ylim(len(contrasts) - 0.5, -0.5)
    pf.panel_label(axes[row][0], "ab"[row], dx_pt=-130)
pf.save(fig, str(out / "human_routes"))

if comp is not None and comp.height:
    h = min(pf.MAX_HEIGHT_MM, 30 + 2 * row_h * len(contrasts) + 12)
    fig, axes = pf.figure(pf.TWO_COLUMN_MM, h, ncols=len(routes), sharey=True, squeeze=False)
    key = [Line2D([], [], linestyle="none", marker="^", color="black", markersize=4, label="study's up list"),
           Line2D([], [], linestyle="none", marker="v", color="black", markersize=4, label="study's down list"),
           Line2D([], [], color=pf.GREY, linewidth=3, solid_capstyle="butt", label="middle 95 % of random gene sets"),
           Line2D([], [], linestyle="none", marker="|", color="black", markersize=5, label="median of random sets")]
    fig.legend(handles=key, loc="outside upper center", ncol=4, frameon=False)
    for j, r in enumerate(routes):
        ax = axes[0][j]
        for t in comp.filter(pl.col("route") == r).iter_rows(named=True):
            if t["contrast"] not in y or not t["list_genes"]:
                continue
            yy = y[t["contrast"]] + (-0.2 if t["direction"] == "up" else 0.2)
            ax.plot([t["null_lo"], t["null_hi"]], [yy, yy], color=pf.GREY, linewidth=2.5, solid_capstyle="butt", zorder=1)
            ax.plot([t["null_median"]], [yy], marker="|", color="black", markersize=4, zorder=2)
            ax.plot([t["share"]], [yy], marker=MARK[t["direction"]], color=COL[r], markersize=3.5,
                    linestyle="none", markeredgewidth=0, zorder=3)
        ax.set_title(ROUTES[r])
        ax.set_xlabel("share of the study's\nlist reached")
        ax.set_xlim(0, 1)
        ax.grid(axis="x", color="#dddddd", linewidth=0.4)
    axes[0][0].set_yticks(range(len(contrasts)))
    axes[0][0].set_yticklabels(labels)
    axes[0][0].set_ylim(len(contrasts) - 0.5, -0.5)
    pf.save(fig, str(out / "human_recovery"))
print("wrote", out)
