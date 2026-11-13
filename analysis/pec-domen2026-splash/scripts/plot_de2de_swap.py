#!/usr/bin/env python3
"""Label-swap control of the DE-to-DE test (notebook 361, design/de2de/README.md).

Usage: plot_de2de_swap.py <real de2de_results.tsv> <swap de2de_results.tsv> <out_stem>

Writes <out_stem>.{png,pdf,svg} and <out_stem>.tsv. One panel per human series, one row per
Botryllus contrast and setting. Coloured mark: D with the real Botryllus labels (filled = pass,
open = fail, grey x = too few subjects). Black ticks: D of each label-swap set of that contrast.
The table gives, for every real pass, whether it holds under the rule in design/de2de/README.md.
"""
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "notebooks"))
import pubfig as pf  # noqa: E402

real_tsv, swap_tsv, out_stem = sys.argv[1:]
C = pf.OKABE_ITO
real = pl.read_csv(real_tsv, separator="\t")
swap = (pl.read_csv(swap_tsv, separator="\t")
        .with_columns(pl.col("botryllus").str.split("__").list.get(0).alias("contrast"),
                      pl.col("botryllus").str.split("__").list.get(1).alias("swap")))
BOT = {"young_13_49d": "young 13-49 d", "young_132d": "young 132 d", "old_41_93d": "old 41-93 d"}
SET = {"protein": ("protein k7", C["blue"], "o"), "dayhoff": ("Dayhoff k13", C["orange"], "s"),
       "hp": ("HP k25", C["reddish_purple"], "D")}
HUMAN = {"GSE99963": "GSE99963\nafter 10 days vs before", "GSE144304": "GSE144304\nfit old vs young",
         "GSE120862": "GSE120862\ntrained vs untrained"}

# The rule, for every real pass
rows = []
for r in real.filter(pl.col("verdict") == "pass").iter_rows(named=True):
    s = swap.filter((pl.col("contrast") == r["botryllus"]) & (pl.col("alphabet") == r["alphabet"])
                    & (pl.col("human") == r["human"]))
    bigger = int((s["D"].abs() >= abs(r["D"])).sum())
    same_sign_pass = int(((s["verdict"] == "pass") & ((s["D"] > 0) == (r["D"] > 0))).sum())
    rows.append(dict(botryllus=r["botryllus"], alphabet=r["alphabet"], human=r["human"], real_D=r["D"],
                     real_p=r["p"], swaps=s.height, swap_D=",".join(f"{d:+.4f}" for d in s["D"]),
                     swap_p=",".join(f"{p:.2g}" for p in s["p"]), swaps_with_larger_abs_D=bigger,
                     swaps_passing_same_sign=same_sign_pass,
                     verdict="holds" if bigger == 0 and same_sign_pass == 0 else "does not hold"))
tab = pl.DataFrame(rows)
tab.write_csv(f"{out_stem}.tsv", separator="\t")
with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=250):
    print(tab)
    print(swap.select("botryllus", "alphabet", "human", "up_kept", "down_kept", "D", "p", "verdict"))

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
        r = real.filter((pl.col("botryllus") == b) & (pl.col("alphabet") == a) & (pl.col("human") == h)).row(0, named=True)
        s = swap.filter((pl.col("contrast") == b) & (pl.col("alphabet") == a) & (pl.col("human") == h))
        ax.scatter(s["D"].to_numpy(), [yy] * s.height, marker="|", color="black", s=30, linewidths=0.8, zorder=2)
        lab, col, mk = SET[a]
        if r["verdict"] == "too few subjects to decide":
            ax.scatter([r["D"]], [yy], marker="x", color="0.45", s=12, linewidths=0.7, zorder=3)
        else:
            ax.scatter([r["D"]], [yy], marker=mk, s=14, facecolors=col if r["verdict"] == "pass" else "white",
                       edgecolors=col, linewidths=0.7, zorder=3)
    ax.axvline(0, color="0.45", linewidth=0.6, linestyle=":", zorder=0)
    ax.set_xlabel("D: human change of up k-mers\nminus down k-mers (log2)")
    ax.set_title(title)
axs[0].set_yticks(list(ys.values()), labels)
handles = [pf.plt.Line2D([], [], marker="o", color=C["blue"], linestyle="none", markersize=3,
                         label="real labels, pass p ≤ 0.05/27"),
           pf.plt.Line2D([], [], marker="o", markerfacecolor="white", color=C["blue"], linestyle="none", markersize=3,
                         label="real labels, fail"),
           pf.plt.Line2D([], [], marker="x", color="0.45", linestyle="none", markersize=3,
                         label="real labels, 7 subjects: cannot reach 0.05/27"),
           pf.plt.Line2D([], [], marker="|", color="black", linestyle="none", markersize=5,
                         label="swapped Botryllus labels (3 swaps; 1 for old 41-93 d)")]
fig.legend(handles=handles, loc="outside upper center", ncol=4, frameon=False, fontsize=5)
pf.save(fig, out_stem)
print("wrote", out_stem)
