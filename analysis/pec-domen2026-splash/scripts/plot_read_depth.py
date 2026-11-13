#!/usr/bin/env python3
"""Reads per sample after trimming, by contrast and condition.

Usage: plot_read_depth.py <read_depth_raw.tsv> <out_stem>

Input: contrast, sample, condition, reads (no header), from each sample's fastp JSON
(summary.after_filtering.total_reads, read 1 and read 2 counted separately). Writes
<out_stem>.tsv with per-contrast totals for control and PEC, and the figure.
"""
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "notebooks"))
import pubfig as pf  # noqa: E402

raw, out_stem = sys.argv[1:]
d = pl.read_csv(raw, separator="\t", has_header=False, new_columns=["contrast", "sample", "condition", "reads"])
CONTRASTS = ["young_2h", "young_24h", "old_24h", "young_13_49d", "young_132d", "old_41_93d"]
NAME = {"young_2h": "young 2 h", "young_24h": "young 24 h", "old_24h": "old 24 h",
        "young_13_49d": "young 13-49 d", "young_132d": "young 132 d", "old_41_93d": "old 41-93 d"}
tot = (d.group_by("contrast", "condition").agg(pl.len().alias("samples"), pl.col("reads").sum().alias("reads"))
       .pivot(on="condition", index="contrast", values=["samples", "reads"])
       .with_columns((pl.col("reads_control") / pl.col("reads_PEC")).round(2).alias("control_over_PEC")))
tot = tot.sort(pl.col("contrast").replace_strict({c: i for i, c in enumerate(CONTRASTS)}))
tot.write_csv(f"{out_stem}.tsv", separator="\t")
print(tot)

pf.use_style()
fig, ax = pf.figure(pf.ONE_COLUMN_MM, 60)
for i, c in enumerate(CONTRASTS):
    for j, (cond, face) in enumerate([("control", "black"), ("PEC", "white")]):
        x = d.filter((pl.col("contrast") == c) & (pl.col("condition") == cond))
        ax.scatter([i + (j - 0.5) * 0.3] * x.height, x["reads"] / 1e6, s=10, facecolors=face,
                   edgecolors="black", linewidths=0.6, label=cond if i == 0 else None)
    r = tot.filter(pl.col("contrast") == c).row(0, named=True)
    ax.text(i, 400, f"{r['reads_control'] / 1e6:.0f}\n{r['reads_PEC'] / 1e6:.0f}", ha="center", va="bottom", fontsize=5)
ax.set_yscale("log")
ax.set_ylim(0.05, 1500)
ax.set_xticks(range(len(CONTRASTS)), [NAME[c] for c in CONTRASTS], rotation=30, ha="right")
ax.set_ylabel("reads after trimming (millions)")
ax.legend(loc="lower left", frameon=False, title="text above: total control /\ntotal PEC (millions)")
pf.save(fig, out_stem)
print("wrote", out_stem)
