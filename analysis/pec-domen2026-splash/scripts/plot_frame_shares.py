#!/usr/bin/env python3
"""Plot the share of each reading frame's k-mers found in Swiss-Prot, by frame, for Botryllus reads.

Usage: plot_frame_shares.py <exports/frame_shares dir> <out_stem>

Reads <dir>/frame_shares.{protein.k7,dayhoff.k12}_hist.tsv from frame_shares.py.
Rows: protein 7-mers, Dayhoff 12-mers (orpheum's two defaults). Columns: simulated reads cut from
CDS, real reads wholly in a CDS. Each line is, for one group of frames (the true
frame, the two other frames on the same strand, the three frames on the opposite strand), the share
of those frames whose Swiss-Prot share is at least x. A 150 nt read has 44 protein 7-mer windows,
so shares come in steps of 1/44; a histogram with 0.02 bins zigzags, this curve does not. Dashed vertical lines: the search's two cutoffs, 0.1 and 0.5.
"""
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "notebooks"))
import pubfig as pf  # noqa: E402

d, out_stem = Path(sys.argv[1]), sys.argv[2]
SETTINGS = [("protein.k7", "protein 7-mers"), ("dayhoff.k12", "Dayhoff 12-mers")]
SOURCES = ["simulated reads cut from CDS", "real reads wholly in a CDS"]
GROUPS = {  # group: (colour, line style)
    "true frame": ("black", "-"),
    "same strand, 1 nt off": ("#888888", "-"),
    "same strand, 2 nt off": ("#888888", "--"),
    "opposite strand (3 frames)": ("#888888", ":"),
}

pf.use_style()
fig, axs = pf.figure(pf.TWO_COLUMN_MM, 95, nrows=2, ncols=2, sharex=True, layout="constrained")
for i, (tag, tname) in enumerate(SETTINGS):
    h = pl.read_csv(d / f"frame_shares.{tag}_hist.tsv", separator="\t")
    for j, src in enumerate(SOURCES):
        ax = axs[i][j]
        for g, (col, ls) in GROUPS.items():
            t = (h.filter((pl.col("source") == src) & (pl.col("group") == g))
                 .group_by("bin").agg(pl.col("frames").sum()).sort("bin"))
            at_least = t["frames"].reverse().cum_sum().reverse() / t["frames"].sum()
            ax.step(t["bin"], at_least, where="post", color=col, linestyle=ls, linewidth=0.9, label=g)
        for c in (0.1, 0.5):
            ax.axvline(c, color="#cccccc", linewidth=0.6, zorder=0)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_title(f"{src}, {tname}")
        if j == 0:
            ax.set_ylabel("share of frames with\nat least this share")
        if i == 1:
            ax.set_xlabel("share of the frame's k-mers found in Swiss-Prot (x)")
for a, l in zip(axs.flat, "abcd"):
    pf.panel_label(a, l)
fig.legend(*axs[0][0].get_legend_handles_labels(), loc="outside upper center", ncols=4, frameon=False,
           title="frame of the read, relative to its true frame (light grey vertical lines: cutoffs 0.1 and 0.5)")
pf.save(fig, out_stem)
print("wrote", out_stem)
