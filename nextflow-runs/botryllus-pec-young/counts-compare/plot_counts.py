#!/usr/bin/env python3
"""Figure for counts-compare/README.md.

Usage: plot_counts.py <counts-compare dir>

a. How well two count tables agree on each gene's change with treatment: the median of the
   paper's four comparisons (dot) and their range (line), one row per pair of tables.
b. Reads the paper's table places in genes divided by reads this run's Salmon table places in
   genes, one dot per library, one row per comparison.
Reads change_per_comparison.tsv and reads_in_genes_per_library.tsv; writes
figures/counts_compare_domen2026_dataset_s1.{png,svg} next to the other figures of this run.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "notebooks"))
import pubfig as pf  # noqa: E402

here = Path(sys.argv[1])
changes = pd.read_csv(here / "change_per_comparison.tsv", sep="\t")
changes = changes[changes.genes_compared == "all genes"]
totals = pd.read_csv(here / "reads_in_genes_per_library.tsv", sep="\t")
# One colour for the pair that is the gap itself, in both panels; grey for the other pairs.
GAP = pf.OKABE_ITO["vermillion"]
OTHER = "0.45"
GAP_PAIR = "D. paper against salmon"
ROWS = {
    "B2. salmon against rsem": "Salmon vs RSEM\n(two standard counters, same genes)",
    "C. no-genome on the paper's genes against no-genome on this run's genes": "Paper's genes vs this run's genes\n(one counter used on both)",
    "B. salmon against no-genome, on this run's genes": "Salmon vs the no-genome counter\n(this run's genes)",
    "A. paper against no-genome, on the paper's genes": "Paper's HTSeq vs the no-genome counter\n(the paper's genes)",
    GAP_PAIR: "Paper's table vs this run's Salmon\n(everything differs: the gap)",
}
COMPARISONS = ["2 h", "24 h", "13-49 d", "132 d"]

pf.use_style()
fig, (a, b) = pf.figure(pf.TWO_COLUMN_MM, 68, ncols=2, layout="constrained", gridspec_kw=dict(width_ratios=[1.2, 1]))

reference = changes[changes.pair == "B2. salmon against rsem"].rank_correlation_of_change.median()
a.axvline(reference, color=OTHER, lw=0.5, ls="--", zorder=1)
a.text(reference, -0.75, "two standard counters", ha="center", va="bottom", color=OTHER,
       bbox=dict(facecolor="white", edgecolor="none", pad=1))
for y, (pair, label) in enumerate(ROWS.items()):
    v = changes[changes.pair == pair].rank_correlation_of_change
    assert len(v) == len(COMPARISONS), pair
    c = GAP if pair == GAP_PAIR else OTHER
    a.plot([v.min(), v.max()], [y, y], color=c, lw=1.5, solid_capstyle="butt")
    a.plot(v.median(), y, "o", color=c, markersize=5, zorder=3)
    a.annotate(f"{v.median():.2f}", (v.max(), y), xytext=(5, 0), textcoords="offset points", va="center", color=c)
a.set_yticks(range(len(ROWS)), list(ROWS.values()))
a.set_ylim(len(ROWS) - 0.5, -1.0)
a.set_xlim(0.5, 0.95)
a.set_xlabel("Agreement on each gene's change with treatment\n(rank correlation; dot, median of the 4 comparisons; line, their range)")

rng = np.random.default_rng(0)  # only for the up-and-down spread of the dots
for y, comparison in enumerate(COMPARISONS):
    r = totals[totals.comparison == comparison].paper_over_salmon
    b.scatter(r, y + rng.uniform(-0.18, 0.18, len(r)), s=12, color=GAP, zorder=3)
    m = r.median()
    b.plot([m, m], [y - 0.3, y + 0.3], color="black", lw=1)
    b.annotate(f"median {m:.2f}  (n = {len(r)})", (10.5, y), ha="left", va="center")
b.axvline(1, color=OTHER, lw=0.5, ls="--", zorder=1)
b.text(1, -0.75, "same reads in genes", ha="left", va="bottom", color=OTHER,
       bbox=dict(facecolor="white", edgecolor="none", pad=1))
b.set_xscale("log")
b.set_xticks([0.5, 1, 2, 5, 10], ["0.5", "1", "2", "5", "10"])
b.minorticks_off()
b.set_xlim(0.5, 40)
b.set_yticks(range(len(COMPARISONS)), COMPARISONS)
b.set_ylim(len(COMPARISONS) - 0.5, -1.0)
b.set_ylabel("Paper's comparison")
b.set_xlabel("Reads in genes, paper's table ÷ this run's Salmon\n(one dot per library; tick, median)")
b.spines["bottom"].set_bounds(0.5, 10)

a.plot([], [], "o-", color=GAP, markersize=5, label="paper's table vs this run's Salmon")
a.plot([], [], "o-", color=OTHER, markersize=5, label="other pairs of count tables")
pf.shared_legend(fig)
pf.panel_label(a, "a")
pf.panel_label(b, "b")
pf.save(fig, str(here.parent / "figures" / "counts_compare_domen2026_dataset_s1"))
