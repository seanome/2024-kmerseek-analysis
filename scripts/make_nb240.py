#!/usr/bin/env python3
"""Generate notebooks/240_alphabet_dose_response_pfam_membrane_heldout.ipynb (experiment 7)."""

import json
from pathlib import Path

cells = []


def md(source):
    cells.append({"cell_type": "markdown", "id": f"md-{len(cells):02d}", "metadata": {},
                  "source": source.strip().splitlines(keepends=True)})


def code(source):
    cells.append({"cell_type": "code", "id": f"code-{len(cells):02d}", "execution_count": None,
                  "metadata": {"jupyter": {"source_hidden": True}}, "outputs": [],
                  "source": source.strip("\n").splitlines(keepends=True)})


md(r"""
# 240. The alphabet result as its own finding: dose-response on Pfam with membrane instances held out (experiment 7)

nb 234: HP retention per substitution sits on the BLOSUM62 line, so the alphabet does not
discover conservation the aligners lack. The midi-plus report: a 2-letter alphabet
recognises the right family (family_fmax) about 1.7x better than protein20 on the same
engine on Pfam. Together that is a methods finding about what reduced alphabets buy: the
same conservation, at 2 bits per residue, with an index and no matrix. This notebook
redoes the dose-response with every Pfam instance that is really a transmembrane segment
held out (150 of 2,435 human instances: at least half the interval under Swiss-Prot
TRANSMEM/INTRAMEM features, or two or more helices inside it), so nb 231's artifact
cannot be what drives it.

Input: `scripts/rescore_pfam_membrane_heldout.py` over the midi-plus Pfam call tables
(`dedup`), heldout split, 9 species. The x-axis is bits per k-mer (k x log2 classes), the
report's "choose bits, not letters" axis.
""")

code(r"""
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

sys.path.insert(0, str(Path.cwd()))
import hp_conservation_utils as hc

pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_width_chars(220)
FIG = Path("../figures")
P = Path("/Users/olga/data/qfo-pfam-region-midi-plus/240_pfam_membrane_heldout_fmax.parquet")
if not P.exists():
    raise SystemExit("Pull the Pfam call tables from Sherlock and run scripts/rescore_pfam_membrane_heldout.py first (see its docstring).")
d = pl.read_parquet(P).filter(pl.col("species_mya").is_not_null() & (pl.col("split") == "heldout"))
km = d.filter(pl.col("tool") == "kmerseek").with_columns(
    pl.col("variant").str.extract(r"^(.+)_k\d+", 1).alias("alphabet"), pl.col("variant").str.extract(r"_k(\d+)", 1).cast(pl.Int64).alias("k"))
km = km.with_columns(pl.col("alphabet").map_elements(lambda a: int(re.search(r"(\d+)$", a).group(1)) if re.search(r"(\d+)$", a) else 20, return_dtype=pl.Int64).alias("classes"))
km = km.with_columns((pl.col("k") * pl.col("classes").cast(pl.Float64).log(2)).alias("bits"))
mean = km.group_by("subset", "alphabet", "k", "classes", "bits").agg(pl.col("fmax").mean(), pl.col("family_fmax").mean(), pl.len().alias("n_species"))
print(mean.filter(pl.col("subset") == "non_membrane").sort("family_fmax", descending=True).head(10))
""")

md(r"""
## 1. Recognition against placement, per alphabet, all instances vs membrane held out
""")

code(r"""
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True, sharex=True)
cmap = plt.cm.viridis
for ax, subset in zip(axes, ["all", "non_membrane"]):
    s = mean.filter(pl.col("subset") == subset)
    for a in sorted(s["alphabet"].unique().to_list(), key=lambda x: -s.filter(pl.col("alphabet") == x)["classes"][0]):
        t = s.filter(pl.col("alphabet") == a).sort("k")
        c = cmap(np.log2(t["classes"][0]) / np.log2(20))
        ax.plot(t["fmax"], t["family_fmax"], marker="o", ms=3, color=c, lw=1, label=f"{a} ({t['classes'][0]})")
    ax.plot([0, 0.5], [0, 0.5], color="0.6", lw=0.8)
    ax.set_xlabel("Fmax (interval: right family, right place)")
    ax.set_title("all Pfam instances" if subset == "all" else "membrane instances held out", fontsize=10.5)
    ax.grid(alpha=0.3)
axes[0].set_ylabel("family Fmax (right family, placement ignored)")
axes[1].legend(fontsize=6.5, ncol=2)
best = {}
for subset in ["all", "non_membrane"]:
    s = mean.filter(pl.col("subset") == subset)
    for cls in [2, 4, 20]:
        b = s.filter(pl.col("classes") == cls).sort("family_fmax", descending=True).row(0, named=True)
        best[(subset, cls)] = b
hc.finish_figure(
    fig, FIG / "240_dose_response_membrane_heldout.png",
    tools="kmerseek, every alphabet and k of the midi-plus sweep (dedup, Pfam heldout split, 9-proteome mean); no comparison tool",
    hypothesis="The 2-letter alphabets' advantage in family recognition over protein20 survives holding out the transmembrane Pfam instances.",
    conclusion=(f"Best family_fmax, all instances: 2-letter {best[('all', 2)]['alphabet']} k{best[('all', 2)]['k']} {best[('all', 2)]['family_fmax']:.3f} vs protein20 k{best[('all', 20)]['k']} {best[('all', 20)]['family_fmax']:.3f} "
                f"(ratio {best[('all', 2)]['family_fmax'] / best[('all', 20)]['family_fmax']:.2f}). Membrane held out: {best[('non_membrane', 2)]['alphabet']} k{best[('non_membrane', 2)]['k']} {best[('non_membrane', 2)]['family_fmax']:.3f} vs protein20 {best[('non_membrane', 20)]['family_fmax']:.3f} "
                f"(ratio {best[('non_membrane', 2)]['family_fmax'] / best[('non_membrane', 20)]['family_fmax']:.2f})."),
    title="Recognition vs placement, by alphabet and k",
)
""")

md(r"""
## 2. Family recognition against bits per k-mer
""")

code(r"""
fig, ax = plt.subplots(figsize=(9, 4.6))
s = mean.filter(pl.col("subset") == "non_membrane")
for a in sorted(s["alphabet"].unique().to_list(), key=lambda x: -s.filter(pl.col("alphabet") == x)["classes"][0]):
    t = s.filter(pl.col("alphabet") == a).sort("bits")
    ax.plot(t["bits"], t["family_fmax"], marker="o", ms=3, lw=1, color=cmap(np.log2(t["classes"][0]) / np.log2(20)), label=f"{a} ({t['classes'][0]})")
ax.axvspan(23, 32, color="0.85", alpha=0.6, lw=0)
ax.set_xlabel("bits per k-mer (k x log2 classes)")
ax.set_ylabel("family Fmax, membrane instances held out")
ax.legend(fontsize=6.5, ncol=3)
ax.grid(alpha=0.3)
peak = s.sort("family_fmax", descending=True).row(0, named=True)
hc.finish_figure(
    fig, FIG / "240_family_fmax_vs_bits.png",
    tools="kmerseek, every alphabet and k of the midi-plus sweep (dedup, Pfam heldout split, membrane instances held out); no comparison tool",
    hypothesis="At matched bits per k-mer, fewer classes recognise the family better: the reduced alphabet buys recognition, not placement.",
    conclusion=f"Peak family Fmax with membrane held out: {peak['alphabet']} k{peak['k']} at {peak['bits']:.0f} bits, {peak['family_fmax']:.3f}. The grey band is the report's 23-32 bit winners' band.",
    title="Family recognition vs bits per k-mer",
)
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "2025-kmerseek-analysis", "language": "python", "name": "2025-kmerseek-analysis"}, "language_info": {"name": "python", "version": "3.12"}}, "nbformat": 4, "nbformat_minor": 5}
out = Path(__file__).resolve().parents[1] / "notebooks" / "240_alphabet_dose_response_pfam_membrane_heldout.ipynb"
out.write_text(json.dumps(nb, indent=1))
print(f"wrote {out} ({len(cells)} cells)")
