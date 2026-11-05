#!/usr/bin/env python3
"""gene-counts and DESeq2 under real and swapped labels (notebook 361).

Usage: plot_gene_counts_swaps.py <swaps.tsv> <out_stem>

One row per contrast. a. Genes PyDESeq2 keeps at p <= 0.05 and 2-fold. b. Of those genes, the share
gene-counts (mean of samples) gives the same direction, on the same labels. Black diamond: real labels.
Open circle: one swapped design, which flips some but not all counted pairs.
Writes <out_stem>.tsv with the totals, and prints it.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "notebooks"))
import pubfig as pf  # noqa: E402

swaps_tsv, out_stem = sys.argv[1:]
NAMES = {"young_2h": "young 2 h", "young_24h": "young 24 h", "old_24h": "old 24 h", "young_13_49d": "young 13-49 d",
         "young_132d": "young 132 d", "old_41_93d": "old 41-93 d"}
t = pd.read_csv(swaps_tsv, sep="\t")
t["pct_agree"] = 100 * t.agree / t.deseq2_genes
g = t.groupby("swapped")[["deseq2_genes", "agree"]].sum()
g["pct_agree"] = 100 * g.agree / g.deseq2_genes
g["designs"] = t.groupby("swapped").size()
g.round(1).to_csv(f"{out_stem}.tsv", sep="\t")
print(g.round(1))

pf.use_style()
fig, axs = pf.figure(pf.ONE_COLUMN_MM * 1.6, 62, ncols=2, sharey=True, layout="constrained")
ys = dict(zip(NAMES, range(len(NAMES))[::-1]))
for ax, col in zip(axs, ("deseq2_genes", "pct_agree")):
    for y in ys.values():
        ax.axhline(y, color="0.9", linewidth=0.5, zorder=0)
    s, r = t[t.swapped], t[~t.swapped]
    ax.plot(s[col], s.contrast.map(ys), linestyle="none", marker="o", markersize=3.5, markerfacecolor="white", markeredgecolor="0.3",
            markeredgewidth=0.6, label="swapped labels, one design", zorder=2)
    ax.plot(r[col], r.contrast.map(ys), linestyle="none", marker="D", markersize=3.5, color="black", label="real labels", zorder=3)
axs[0].set_yticks(list(ys.values()), list(NAMES.values()))
axs[0].set_xscale("log")
axs[0].set_xticks([1, 10, 100, 1000], ["1", "10", "100", "1,000"])
axs[0].minorticks_off()
axs[0].set_xlabel("genes DESeq2 keeps\n(p ≤ 0.05, 2-fold)")
axs[1].set_xlim(0, 100)
axs[1].set_xlabel("of those, gene-counts gives\nthe same direction (%)")
for a, l in zip(axs, "ab"):
    pf.panel_label(a, l)
fig.legend(*axs[0].get_legend_handles_labels(), loc="outside upper center", ncols=2, frameon=False)
pf.save(fig, out_stem)
print("wrote", out_stem)
