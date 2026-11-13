#!/usr/bin/env python3
"""Plot how often each way of choosing a read's reading frame is right, for Botryllus reads.

Usage: plot_frame_methods.py <frame_methods.tsv> <out_stem>

One bar per method (frame_methods.py), split into: the true frame chosen alone (right), the true
frame tied with a wrong one, a wrong frame, and no frame chosen. Left panel: simulated reads cut
from new-genome CDS (coding sequences); right panel: real reads whose true frame comes from a
DIAMOND blastx hit to a new-genome protein. The number after each bar is the right share.
Methods against half of the new-genome genes are scored only on reads from the other half.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "notebooks"))
import pubfig as pf  # noqa: E402

t = pl.read_csv(sys.argv[1], separator="\t")
out_stem = sys.argv[2]

REFS = [
    ("swissprot", "k-mers in Swiss-Prot"),
    ("botznik", "k-mers in old Botryllus gene models"),
    ("swissprot_botznik", "k-mers in Swiss-Prot + old gene models"),
    ("newgenome_halfA", "k-mers in half the new-genome genes (read's own gene left out)"),
    ("newgenome", "k-mers in the new-genome proteins (read's own gene in it: best case)"),
]
SETTINGS = ["protein.k5", "protein.k6", "protein.k7", "protein.k10",
            "dayhoff.k9", "dayhoff.k11", "dayhoff.k12", "dayhoff.k18"]


def setting_label(s):
    a, k = s.split(".k")
    return f"{'protein' if a == 'protein' else 'Dayhoff'} {k}"


groups = [("Stop codons only", [("stops.one", "the one stop-free frame"),
                                ("stops.longest", "longest stop-free stretch")])]
for ref, title in REFS:
    groups.append((title, [(f"{ref}.{s}", setting_label(s)) for s in SETTINGS]))
groups.append(("DIAMOND blastx to old gene models", [("blastx.fast", "--fast"), ("blastx.default", "default")]))
combo = [m for m in t["method"].unique().to_list() if "+" in m]


def combo_label(m):
    parts = []
    for p in m.split("+"):
        if p.startswith("blastx."):
            parts.append(f"blastx {p.split('.')[1]}")
        elif p == "stops.one1":
            parts.append("one stop-free frame")
        elif p == "stops.longest":
            parts.append("longest stretch")
        else:
            parts.append("old-model " + setting_label(p.split(".", 1)[1]))
    return ", then ".join(parts)


groups.append(("Combinations, first that gives a frame", [(m, combo_label(m)) for m in sorted(combo)]))

rows = []  # (y label, method) top to bottom, with group headers
for title, ms in groups:
    ms = [(m, lab) for m, lab in ms if m in set(t["method"])]
    if ms:
        rows.append((title, None))
        rows.extend((lab, m) for m, lab in ms)

CATS = [("right", "true frame alone", "#222222", None),
        ("tied", "true frame tied with a wrong one", "#9a9a9a", None),
        ("wrong", "wrong frame", pf.OKABE_ITO["yellow"], None),
        ("none", "no frame", "white", None)]

pf.use_style()
h = 4 + 2.3 * len(rows)
fig, axes = pf.figure(pf.TWO_COLUMN_MM, min(h, pf.MAX_HEIGHT_MM), ncols=2, sharey=True)
for ax, src, title in zip(axes, ["simulated", "real"],
                          ["Simulated reads cut from new-genome CDS", "Real reads, frame from blastx to new-genome proteins"]):
    y = list(range(len(rows)))[::-1]
    for yi, (lab, m) in zip(y, rows):
        if m is None:
            continue
        r = t.filter((pl.col("method") == m) & (pl.col("reads") == src))
        if r.height == 0:
            continue
        r = r.row(0, named=True)
        left = 0.0
        for c, name, col, _ in CATS:
            ax.barh(yi, r[c], left=left, color=col, edgecolor="#555555", linewidth=0.3, height=0.75,
                    label=name)
            left += r[c]
        ax.text(1.01, yi, f"{100 * r['right']:.0f} %", va="center", ha="left", fontsize=5,
                transform=ax.get_yaxis_transform())
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xticklabels(["0", "25", "50", "75", "100 %"])
    ax.set_xlabel("share of reads")
    ax.set_title(title, fontsize=6)
    ax.tick_params(axis="y", length=0)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
axes[0].set_yticks(list(range(len(rows)))[::-1])
axes[0].set_yticklabels([lab for lab, _ in rows])
for tick, (lab, m) in zip(axes[0].get_yticklabels(), rows):
    if m is None:
        tick.set_fontweight("bold")
pf.shared_legend(fig, ncol=4, frameon=False)
pf.save(fig, out_stem)
print("rows", len(rows), "height mm", min(h, pf.MAX_HEIGHT_MM))
