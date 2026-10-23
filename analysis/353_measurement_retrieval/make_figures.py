#!/usr/bin/env python3
"""Figure for notebook 353: what the evidence collection holds.

a  ELM instances by the curators' verdict.
b  Sanborn et al. 2021 activation tiles by Z score, with the paper's significance cut-off
   (one-sided p < 0.0001, Z >= 3.719) and p < 0.05 (Z >= 1.645).

Usage: python make_figures.py <work folder>   (reads <work>/evidence/evidence.parquet)
"""

import sys
from pathlib import Path

import numpy as np
import polars as pl
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "notebooks"))
import pubfig as pf  # noqa: E402

# One colour per meaning, in both panels.
WORKS = pf.OKABE_ITO["blue"]
WEAK = pf.OKABE_ITO["orange"]
DOES_NOT = pf.OKABE_ITO["vermillion"]
NO_VERDICT = pf.GREY

ev = pl.read_parquet(Path(sys.argv[1]) / "evidence" / "evidence.parquet")
elm = ev.filter(pl.col("source") == "ELM")["outcome"].value_counts()
elm_n = dict(zip(elm["outcome"], elm["count"]))
tiles = ev.filter(pl.col("source") == "Sanborn2021")
z = tiles["z_score"].drop_nulls().to_numpy()
n_unmeasured = tiles["z_score"].null_count()
lab = dict(zip(*tiles["outcome"].value_counts().to_dict(as_series=False).values()))

pf.use_style()
fig, (a, b) = pf.figure(
    pf.TWO_COLUMN_MM, 72, ncols=2, gridspec_kw={"width_ratios": [1, 1.6]}
)

order = [
    ("true positive", WORKS),
    ("true negative", DOES_NOT),
    ("false positive", DOES_NOT),
    ("unknown", NO_VERDICT),
]
y = np.arange(len(order))[::-1]
for yi, (name, col) in zip(y, order):
    a.barh(yi, elm_n.get(name, 0), color=col, height=0.6)
    a.text(elm_n.get(name, 0) * 1.15, yi, f"{elm_n.get(name, 0):,}", va="center")
a.set_xscale("log")
a.set_xlim(10, 20000)
a.set_xticks([10, 100, 1000, 10000], ["10", "100", "1,000", "10,000"])
a.set_yticks(y, [n for n, _ in order])
a.set_xlabel("ELM instances (n); bar length by ratio")
a.set_title(f"ELM, {sum(elm_n.values()):,} instances, curators' verdict", loc="left")

# Bin edges fall on the two cut-offs, so no bar straddles one.
w = 0.25
for lo, hi, colour in (
    (np.floor(z.min()), 1.645, DOES_NOT),
    (1.645, 3.719, WEAK),
    (3.719, np.ceil(z.max()) + w, WORKS),
):
    edges = (
        np.append(np.arange(hi, lo - w, -w)[::-1], hi)
        if colour is DOES_NOT
        else np.arange(lo, hi + w, w)
    )
    edges = np.unique(np.clip(edges, lo, hi))
    sel = z[(z >= lo) & (z < hi)]
    b.hist(sel, bins=edges, color=colour)
cnt = np.histogram(z, np.arange(np.floor(z.min()), z.max() + w, w))[0]
b.set_yscale("log")
b.set_ylim(0.8, cnt.max() * 1.6)
b.set_yticks([1, 10, 100, 1000], ["1", "10", "100", "1,000"])
for x in (1.645, 3.719):
    b.axvline(x, color="black", lw=0.5, ls=(0, (2, 2)))
b.text(3.719 + 0.1, cnt.max(), "paper's cut-off,\np < 0.0001", va="top")
b.text(1.645 + 0.1, cnt.max() / 8, "p < 0.05", va="top")
b.set_xlabel("activation Z score (paper's Z-test; higher = more activation)")
b.set_ylabel("tiles (n)")
b.set_title(
    f"Sanborn 2021, {tiles.height:,} tiles of 53 aa, yeast GFP reporter", loc="left"
)

handles = [
    Patch(
        color=WORKS,
        label=f"works: ELM true positive; tile active ({lab['active']:,} tiles)",
    ),
    Patch(color=WEAK, label=f"tile weak, 0.0001 <= p < 0.05 ({lab['weak']:,} tiles)"),
    Patch(
        color=DOES_NOT,
        label=f"does not work: ELM true negative or false positive; tile inactive ({lab['inactive']:,} tiles)",
    ),
    Patch(
        color=NO_VERDICT,
        label=f"no verdict: ELM unknown; tile not measured ({n_unmeasured} tiles, not plotted)",
    ),
]
fig.legend(handles=handles, loc="outside upper left", ncol=2, frameon=False)
pf.panel_label(a, "a")
pf.panel_label(b, "b")
out = HERE / "figures"
out.mkdir(exist_ok=True)
pf.save(fig, out / "353_evidence_collection_elm_sanborn2021")
print("ELM", elm_n, "tiles", lab, "plotted tiles", int(cnt.sum()), "of", len(z))
