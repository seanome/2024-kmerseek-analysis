#!/usr/bin/env python3
"""Figure for counts-compare/README.md.

Usage: plot_counts.py <counts-compare dir>

a. Rank correlation of the change with treatment between two count tables, per comparison.
b. Reads the paper placed in genes divided by reads this run (Salmon) placed in genes, per library.
Reads change_per_comparison.tsv and reads_in_genes_per_library.tsv; writes
figures/counts_compare_domen2026_dataset_s1.{png,pdf,svg} next to the other figures of this run.
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
C = pf.OKABE_ITO
COLOUR = {"2 h": C["blue"], "24 h": C["vermillion"], "13-49 d": C["bluish_green"], "132 d": C["reddish_purple"]}
ROWS = {
    "B2. salmon against rsem": "Two aligner counts, this run's genes\n(Salmon, RSEM)",
    "C. no-genome on the paper's genes against no-genome on this run's genes": "One counter, the two gene sets\n(no-genome count on each)",
    "B. salmon against no-genome, on this run's genes": "Two counters, this run's genes\n(Salmon, no-genome)",
    "A. paper against no-genome, on the paper's genes": "Two counters, the paper's genes\n(paper's HTSeq, no-genome)",
    "D. paper against salmon": "The gap itself\n(paper's table, this run's Salmon)",
}

pf.use_style()
fig, (a, b) = pf.figure(pf.TWO_COLUMN_MM, 70, ncols=2, layout="constrained", gridspec_kw=dict(width_ratios=[1.25, 1]))
rng = np.random.default_rng(0)  # only for the up-and-down spread of the dots
for y, (pair, label) in enumerate(ROWS.items()):
    d = changes[changes.pair == pair]
    for _, r in d.iterrows():
        a.scatter(
            r.rank_correlation_of_change, y + rng.uniform(-0.15, 0.15), s=14, color=COLOUR[r.comparison], zorder=3
        )
    m = d.rank_correlation_of_change.median()
    a.plot([m, m], [y - 0.32, y + 0.32], color="black", lw=1)
    a.text(1.0, y, f"{m:.2f}", va="center", ha="right", fontsize=6)
a.axvline(0.6, color=pf.GREY, lw=0.5, ls=":")
a.set_yticks(range(len(ROWS)), list(ROWS.values()))
a.set_ylim(len(ROWS) - 0.5, -0.5)
a.set_xlim(0.4, 1.0)
a.set_xlabel("Rank correlation of the change with treatment,\ngene by gene (tick and number, median; dotted, 0.6)")

for comparison, colour in COLOUR.items():
    d = totals[totals.comparison == comparison]
    b.scatter(
        d.trimmed_read_pairs, d.paper_over_salmon, s=14, color=colour, label=f"{comparison} (n = {len(d)})", zorder=3
    )
b.axhline(1, color=pf.GREY, lw=0.5, ls=":")
b.set_xscale("log")
b.set_yscale("log")
b.set_xticks([1e5, 1e6, 1e7, 1e8 / 1.43], ["100,000", "1 million", "10 million", "70 million"])
b.set_yticks([1, 2, 5, 10], ["1 (equal)", "2", "5", "10"])
b.minorticks_off()
b.set_xlabel("Trimmed read pairs in the library")
b.set_ylabel("Reads in genes:\npaper's table / this run's Salmon")
h, l = b.get_legend_handles_labels()
fig.legend(
    h,
    l,
    loc="outside upper center",
    ncol=4,
    frameon=False,
    title="The paper's comparison (one dot per comparison in a, per library in b)",
)
pf.panel_label(a, "a", dx_pt=-150)
pf.panel_label(b, "b", dx_pt=-45)
pf.save(fig, str(here.parent / "figures" / "counts_compare_domen2026_dataset_s1"))
