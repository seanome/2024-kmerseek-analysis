#!/usr/bin/env python3
"""Figure: p-value histograms of the two abundance tests in each contrast.

Usage: plot_power_check.py <power_check.tsv> <power_check_hist.tsv> <out_stem>

One row per contrast, one column per test. Bar height is the share of tests in that 0.05-wide
p-value bin divided by 0.05, so a test with no change anywhere gives bars of height 1 (the
dashed line). Tests where some sequences changed give a tall first bar, as in the simulated
control. The text in each panel gives how many tests were run and how many were kept at
corrected p <= 0.05.
"""
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "notebooks"))
import pubfig as pf  # noqa: E402

summary_tsv, hist_tsv, out_stem = sys.argv[1:]
summary = pl.read_csv(summary_tsv, separator="\t")
hist = pl.read_csv(hist_tsv, separator="\t")

CONTRASTS = ["control", "young_2h", "young_24h", "old_24h", "young_13_49d", "young_132d", "old_41_93d"]
LABEL = {"control": "simulated control\n(400 genes changed)", "young_2h": "young 2 h", "young_24h": "young 24 h",
         "old_24h": "old 24 h", "young_13_49d": "young 13-49 d", "young_132d": "young 132 d",
         "old_41_93d": "old 41-93 d"}
COLUMNS = [("DE-kupl limma, k-mers", "DE-kupl: each 31-letter k-mer"),
           ("anchor counts DESeq2", "DESeq2: count of each SPLASH anchor")]


def method_for(col, contrast):
    if col.startswith("DE-kupl"):
        return col
    # young_132d has no matched pairs, so only the ~ condition model was fitted there
    pair = f"{col} ~ pair + condition"
    have = set(summary.filter(pl.col("contrast") == contrast)["method"])
    return pair if pair in have else f"{col} ~ condition"


pf.use_style()
fig, axs = pf.figure(pf.TWO_COLUMN_MM, 150, nrows=len(CONTRASTS), ncols=len(COLUMNS),
                     sharex=True, sharey=True)
for i, c in enumerate(CONTRASTS):
    for j, (col, title) in enumerate(COLUMNS):
        ax = axs[i, j]
        m = method_for(col, c)
        h = hist.filter((pl.col("contrast") == c) & (pl.col("method") == m)).sort("bin_low")
        rel = h["n"] / h["n"].sum() / 0.05
        ax.bar(h["bin_low"], rel, width=0.05, align="edge", color=pf.GREY if c != "control" else pf.OKABE_ITO["blue"])
        ax.axhline(1, color="black", linestyle="--", linewidth=0.5)
        s = summary.filter((pl.col("contrast") == c) & (pl.col("method") == m)).row(0, named=True)
        design = "" if col.startswith("DE-kupl") else ("  (~ pair + condition)" if "pair" in m else "  (~ condition)")
        ax.text(0.99, 0.95, f"{s['tests']:,} tests, {s['kept_corrected_p_le_0.05']:,} kept{design}",
                transform=ax.transAxes, ha="right", va="top")
        ax.set_ylim(0, 3.3)
        if i == 0:
            ax.set_title(title)
        if j == 0:
            ax.set_ylabel(LABEL[c], rotation=0, ha="right", va="center", labelpad=10)
        if i == len(CONTRASTS) - 1:
            ax.set_xlabel("p-value (before correction)")
fig.supylabel("share of tests in each bin, relative to no change (dashed line = 1)", x=0.0)
pf.save(fig, out_stem)
print("wrote", out_stem)
