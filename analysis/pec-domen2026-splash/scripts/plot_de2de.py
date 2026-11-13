#!/usr/bin/env python3
"""Figure of the DE-to-DE test (notebook 361, design/de2de/README.md).

Usage: plot_de2de.py <de2de_results.tsv> <de2de_null.tsv.gz> <out_stem>

One panel per human series. One row per Botryllus contrast and setting. Grey bar: middle 95 % of D
under shuffled human labels. Mark: real D; filled = pass, open = fail, grey x = too few subjects.
"""
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "notebooks"))
import pubfig as pf  # noqa: E402

res_tsv, null_tsv, out_stem = sys.argv[1:]
C = pf.OKABE_ITO
res = pl.read_csv(res_tsv, separator="\t")
null = pl.read_csv(null_tsv, separator="\t").filter(~pl.col("real"))
BOT = {"young_13_49d": "young 13-49 d", "young_132d": "young 132 d", "old_41_93d": "old 41-93 d"}
SET = {"protein": ("protein k7", C["blue"], "o"), "dayhoff": ("Dayhoff k13", C["orange"], "s"),
       "hp": ("HP k25", C["reddish_purple"], "D")}
HUMAN = {"GSE99963": "GSE99963\nafter 10 days vs before", "GSE144304": "GSE144304\nfit old vs young",
         "GSE120862": "GSE120862\ntrained vs untrained"}

pf.use_style()
fig, axs = pf.figure(pf.TWO_COLUMN_MM, 70, ncols=3, layout="constrained", sharey=True)
ys, labels, y = {}, [], 0
for b in BOT:
    for a in SET:
        ys[b, a] = y
        labels.append(f"{BOT[b]}, {SET[a][0]}")
        y -= 1
    y -= 0.6
for ax, (h, title) in zip(axs, HUMAN.items()):
    for (b, a), yy in ys.items():
        r = res.filter((pl.col("botryllus") == b) & (pl.col("alphabet") == a) & (pl.col("human") == h)).row(0, named=True)
        d = null.filter((pl.col("botryllus") == b) & (pl.col("alphabet") == a) & (pl.col("human") == h))["D"].to_numpy()
        lo, hi = np.quantile(d, [0.025, 0.975])
        ax.plot([lo, hi], [yy, yy], color="0.8", linewidth=3, solid_capstyle="butt", zorder=1)
        lab, col, mk = SET[a]
        if r["verdict"] == "too few subjects to decide":
            ax.scatter([r["D"]], [yy], marker="x", color="0.45", s=12, linewidths=0.7, zorder=3)
        else:
            ax.scatter([r["D"]], [yy], marker=mk, s=14, facecolors=col if r["verdict"] == "pass" else "white",
                       edgecolors=col, linewidths=0.7, zorder=3)
        ax.annotate(f"p {r['p']:.2g}", (1.0, yy), xycoords=("axes fraction", "data"), ha="right", va="center",
                    fontsize=5, color="0.3")
    ax.axvline(0, color="0.45", linewidth=0.6, linestyle=":", zorder=0)
    ax.set_xlim(-0.06, 0.075)
    ax.set_xlabel("D: human change of up k-mers\nminus down k-mers (log2)")
    ax.set_title(title)
axs[0].set_yticks(list(ys.values()), labels)
handles = [pf.plt.Line2D([], [], color="0.8", linewidth=3, label="middle 95 % of shuffled human labels"),
           pf.plt.Line2D([], [], marker="o", color=C["blue"], linestyle="none", markersize=3, label="pass, p ≤ 0.05/27"),
           pf.plt.Line2D([], [], marker="o", markerfacecolor="white", color=C["blue"], linestyle="none", markersize=3,
                         label="fail"),
           pf.plt.Line2D([], [], marker="x", color="0.45", linestyle="none", markersize=3,
                         label="7 subjects: cannot reach 0.05/27")]
fig.legend(handles=handles, loc="outside upper center", ncol=4, frameon=False, fontsize=5)
pf.save(fig, out_stem)
print("wrote", out_stem)
