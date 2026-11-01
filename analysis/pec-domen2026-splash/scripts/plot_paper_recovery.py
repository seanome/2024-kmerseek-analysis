#!/usr/bin/env python3
"""Figure: share of each paper gene list reached, against the baseMean-matched null.

Usage: plot_paper_recovery.py <paper_recovery.tsv> <out_stem>

One row per paper list (contrast and direction). The grey bar is the middle 95 % of 1,000 random
gene sets matched to the list's baseMean deciles, the grey tick their median, and the dot the
share of the real list reached. One panel per channel.
"""
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "notebooks"))
import pubfig as pf  # noqa: E402

tsv, out_stem = sys.argv[1:]
d = pl.read_csv(tsv, separator="\t")
CONTRASTS = ["young_2h", "young_24h", "old_24h", "young_13_49d", "young_132d", "old_41_93d"]
NAME = {"young_2h": "young 2 h", "young_24h": "young 24 h", "old_24h": "old 24 h",
        "young_13_49d": "young 13-49 d", "young_132d": "young 132 d", "old_41_93d": "old 41-93 d"}
CHANNEL = {"splash": "SPLASH anchors (corrected p ≤ 0.05)",
           "anchor_nominal": "anchor counts, DESeq2 (p ≤ 0.05, 2-fold)"}
COLOUR = {"splash": pf.OKABE_ITO["blue"], "anchor_nominal": pf.OKABE_ITO["vermillion"]}
channels = [c for c in CHANNEL if c in set(d["channel"])]
rows = [(c, l) for c in CONTRASTS for l in ("up", "down")]

pf.use_style()
fig, axs = pf.figure(pf.TWO_COLUMN_MM if len(channels) > 1 else pf.ONE_COLUMN_MM, 80,
                     ncols=len(channels), sharey=True, squeeze=False)
for ax, ch in zip(axs[0], channels):
    for y, (c, l) in enumerate(rows):
        r = d.filter((pl.col("channel") == ch) & (pl.col("contrast") == c) & (pl.col("list") == l))
        if r.height == 0:
            continue
        r = r.row(0, named=True)
        ax.plot([r["null_2_5"], r["null_97_5"]], [y, y], color=pf.GREY, linewidth=3, solid_capstyle="butt")
        ax.plot([r["null_median"]], [y], marker="|", color="black", markersize=6)
        ax.plot([r["fraction_reached"]], [y], marker="o", color=COLOUR[ch], markersize=4)
        ax.text(1.02, y, f"{r['reached']}/{r['list_genes_with_protein']}  ({r['share_null_at_or_above']:.3f})",
                transform=ax.get_yaxis_transform(), va="center")
    ax.set_title(CHANNEL[ch])
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of the paper's list reached")
axs[0, 0].set_yticks(range(len(rows)), [f"{NAME[c]}, {l}" for c, l in rows])
axs[0, 0].invert_yaxis()
fig.suptitle("dot: real list   grey bar: middle 95 % of 1,000 baseMean-matched random sets\n"
             "tick: their median   text: reached/listed (share of random sets at or above)")
pf.save(fig, out_stem)
print("wrote", out_stem)
