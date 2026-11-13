"""Redraw supplied Figure 4A/B counts with explicit numeric gene counts."""

import argparse, csv, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "notebooks"))
import pubfig as pf
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("folder", type=Path)
a = p.parse_args()
rows = list(
    csv.DictReader((a.folder / "published-hub-counts.tsv").open(), delimiter="\t")
)
hubs = [
    "DNA / RNA / Epigenetic",
    "Signaling / Regulation",
    "Mitochondrial / Metabolic",
    "Contractile / Cytoskeletal",
    "ECM / Structural",
    "Immune",
]
labels = [
    "DNA / RNA",
    "Signaling / regulation",
    "Mitochondrial / metabolic",
    "Contractile / cytoskeletal",
    "ECM / structural",
    "Immune",
]
settings = [
    ("young", "2 Hours Young", "Young\n2 h"),
    ("young", "24 Hours Young", "Young\n24 h"),
    ("young", "13–49 Days Young", "Young\n13–49 d"),
    ("young", "132 Days Young", "Young\n132 d"),
    ("old", "24 Hours", "Old\n24 h"),
    ("old", "41–96 Days", "Old\n41–96 d"),
]
pf.use_style()
fig, axs = pf.figure(183, 100, ncols=6, sharey=True)
for ax, (cohort, time, title) in zip(axs, settings):
    for direction, offset, color, label in [
        ("Up-reg", -0.18, pf.OKABE_ITO["blue"], "Upregulated"),
        ("Down-reg", 0.18, pf.OKABE_ITO["orange"], "Downregulated"),
    ]:
        vals = [
            int(
                next(
                    r["count"]
                    for r in rows
                    if r["cohort"] == cohort
                    and r["time"] == time
                    and r["hub"] == h
                    and r["direction"] == direction
                )
            )
            for h in hubs
        ]
        y = np.arange(6) + offset
        ax.barh(y, vals, height=0.32, color=color, label=label)
        for yy, n in zip(y, vals):
            ax.text(n + 0.5, yy, str(n), va="center")
    ax.set_xlim(0, 43)
    ax.set_xticks([0, 20, 40])
    ax.set_yticks(range(6), labels)
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_xlabel("Genes (n)")
    ax.text(0.5, 1.02, title, transform=ax.transAxes, ha="center", va="bottom")
axs[0].invert_yaxis()
pf.shared_legend(fig, ncol=2)
# Checklist: color key precedes marks; bar position separates directions; all 72 marks
# have counts; six original functional groups retained; Other is in the source TSV.
pf.save(fig, a.folder / "figure4AB-published-count-redraw")
(a.folder / "figure4AB-published-count-redraw.caption.txt").write_text(
    "Figure 4A/B supplementary counts redrawn as paired bars. Blue upper bars: upregulated genes; orange lower bars: downregulated genes. Numbers are copied from the supplied fig4a&bgene_count sheet, not independently recomputed DE results. The six plotted functional groups omit the supplementary Other category; all 84 original count cells remain in published-hub-counts.tsv. Gene counts do not imply adjusted-P significance.\n"
)
