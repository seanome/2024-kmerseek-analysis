#!/usr/bin/env python3
"""Why the two Botryllus routes' sketch files differ in size (notebook 361).

Usage: plot_sketch_counts.py <sketch_settings.tsv> <contig_lengths.tsv> <out_stem>

a. Zip file size against number of sketches, one point per contrast, route and setting. DNA has one
sketch per contig, the others one per peptide of 20+ residues. b. Contig lengths. A peptide of 20
residues needs 60 bases. Writes <out_stem>.tsv with totals per route and setting, and prints it.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "notebooks"))
import pubfig as pf  # noqa: E402

settings_tsv, lengths_tsv, out_stem = sys.argv[1:]
C = pf.OKABE_ITO
ROUTES = {"splash": ("SPLASH", C["blue"]), "kamrat": ("KaMRaT, t-test, p < 0.05", C["reddish_purple"])}
BYTES = 750
s = pd.read_csv(settings_tsv, sep="\t")
s["dna"] = s.setting.str.startswith("dna")
t = s.groupby(["route", "setting"], as_index=False)[["zip_bytes", "sketches", "hashes"]].sum()
t["bytes_per_sketch"] = (t.zip_bytes / t.sketches).round(1)
t.to_csv(f"{out_stem}.tsv", sep="\t", index=False)
print(t.to_string(index=False))
n = pd.read_csv(lengths_tsv, sep="\t")

pf.use_style()
fig, axs = pf.figure(pf.TWO_COLUMN_MM * 0.75, 66, ncols=2, layout="constrained")
ax = axs[0]
xs = [s.sketches.min() / 2, s.sketches.max() * 2]
ax.plot(xs, [x * BYTES for x in xs], color="0.45", linewidth=0.6, linestyle=":", zorder=1, label=f"{BYTES} bytes per sketch")
for r, (lab, col) in ROUTES.items():
    for dna, kind in ((True, "one sketch per contig"), (False, "one sketch per peptide")):
        d = s[(s.route == r) & (s.dna == dna)]
        ax.plot(d.sketches, d.zip_bytes, linestyle="none", marker="o", markersize=3, markeredgecolor=col, markeredgewidth=0.6,
                markerfacecolor=col if dna else "white", label=f"{lab}, {kind}", zorder=2)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xticks([1e2, 1e3, 1e4, 1e5, 1e6], ["100", "1,000", "10,000", "100,000", "1,000,000"])
ax.set_yticks([1e5, 1e6, 1e7, 1e8, 1e9], ["100 KB", "1 MB", "10 MB", "100 MB", "1 GB"])
ax.minorticks_off()
ax.set_xlabel("sketches in the file")
ax.set_ylabel("file size")
ax = axs[1]
for r, (lab, col) in ROUTES.items():
    d = n[n.route == r]
    ax.step([d.length.min() - 1, *d.length], [0, *(100 * d.contigs.cumsum() / d.contigs.sum())], where="post", color=col, linewidth=1)
ax.axvline(60, color="0.45", linewidth=0.6, linestyle="--", zorder=1, label="60 bases: shortest contig that can hold a peptide")
ax.set_xscale("log")
ax.set_xticks([20, 60, 100, 300, 1000], ["20", "60", "100", "300", "1,000"])
ax.minorticks_off()
ax.set_ylim(0, 102)
ax.set_xlabel("contig length (bases)")
ax.set_ylabel("contigs this long or shorter (%)")
for a, l in zip(axs, "ab"):
    pf.panel_label(a, l)
h = axs[0].get_legend_handles_labels()
h2 = axs[1].get_legend_handles_labels()
fig.legend(h[0] + h2[0], h[1] + h2[1], loc="outside upper center", ncols=2, frameon=False)
pf.save(fig, out_stem)
print("wrote", out_stem)
