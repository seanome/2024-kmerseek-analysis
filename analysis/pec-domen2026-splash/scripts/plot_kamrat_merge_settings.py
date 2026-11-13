#!/usr/bin/env python3
"""Why KaMRaT contigs stay short: merge settings and forks (notebook 361).

Usage: plot_kamrat_merge_settings.py <dir with lengths.*.tsv and young_132d.neighbours.tsv> <out_stem>

a. Share of all contigs long enough to hold a peptide of 20 residues (60+ bases), for each merge
setting, in the three small contrasts. b. Contigs of one k-mer in young 132 d, by what sits next to
them in the k-mer table. Writes <out_stem>.tsv with the numbers in b, and prints it.
"""
import glob
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "notebooks"))
import pubfig as pf  # noqa: E402

in_dir, out_stem = sys.argv[1:]
C = pf.OKABE_ITO["reddish_purple"]
SETTINGS = {"pearson_0.20.ov30-15": "used here: correlation check 0.20, overlap 15-30", "pearson_0.35.ov30-15": "correlation check 0.35, overlap 15-30",
            "pearson_0.50.ov30-15": "correlation check 0.50, overlap 15-30", "none.ov30-15": "no correlation check, overlap 15-30",
            "pearson_0.20.ov30-25": "correlation check 0.20, overlap 25-30", "none.ov30-25": "no correlation check, overlap 25-30"}
CONTRASTS = {"young_13_49d": ("young 13-49 d", "o"), "young_132d": ("young 132 d", "s"), "old_41_93d": ("old 41-93 d", "^")}
t = pd.concat(pd.read_csv(f, sep="\t") for f in glob.glob(f"{in_dir}/lengths.*.tsv"))
n = pd.read_csv(f"{in_dir}/young_132d.neighbours.tsv", sep="\t")
fork = (n.left == "two or more") | (n.right == "two or more")
empty = (n.left == "none") & (n.right == "none")
g = pd.DataFrame(dict(group=["next k-mer missing on both sides", "single next k-mer, no fork", "a fork on at least one side"],
                      contigs=[n.contigs[empty].sum(), n.contigs[~empty & ~fork].sum(), n.contigs[fork].sum()]))
g["pct"] = (100 * g.contigs / g.contigs.sum()).round(1)
g.to_csv(f"{out_stem}.tsv", sep="\t", index=False)
print(g.to_string(index=False))

pf.use_style()
fig, axs = pf.figure(pf.TWO_COLUMN_MM, 62, ncols=2, layout="constrained", gridspec_kw=dict(width_ratios=[1.1, 1]))
ax = axs[0]
ys = dict(zip(SETTINGS, range(len(SETTINGS))[::-1]))
for y in ys.values():
    ax.axhline(y, color="0.9", linewidth=0.5, zorder=0)
for c, (lab, mk) in CONTRASTS.items():
    d = t[t.contrast == c]
    ax.plot(d.all_ge60_pct, d.setting.map(ys), linestyle="none", marker=mk, markersize=3.5, color=C, markerfacecolor="white", markeredgewidth=0.7, label=lab)
ax.set_yticks(list(ys.values()), list(SETTINGS.values()))
ax.set_xlim(0, 100)
ax.set_xlabel("contigs of 60+ bases (%)")
ax.legend(loc="lower right", frameon=False)
ax = axs[1]
ax.barh(g.group[::-1], g.pct[::-1], color=C, height=0.6)
for y, r in enumerate(g[::-1].itertuples()):
    ax.annotate(f"{r.contigs:,} ({r.pct:.1f} %)", (r.pct, y), xytext=(3, 0), textcoords="offset points", va="center", fontsize=5)
ax.set_xlim(0, 100)
ax.set_xlabel("one-k-mer contigs, young 132 d (%)")
for a, l in zip(axs, "ab"):
    pf.panel_label(a, l)
pf.save(fig, out_stem)
print("wrote", out_stem)
