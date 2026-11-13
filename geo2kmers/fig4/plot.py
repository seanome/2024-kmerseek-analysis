"""Draw the Fig. 4C-D test: one panel per comparison, score by group, published AUC beside ours.

Usage: plot.py <results/fig4 folder>
Writes fig4_panels.svg and .png to that folder.
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "notebooks"))
import pubfig as pf

folder = Path(sys.argv[1])


def rows_of(name):
    with open(folder / name) as f:
        return list(csv.DictReader(f, delimiter="\t"))


scores = defaultdict(lambda: {"pre": [], "post": []})
order = []
for r in rows_of("sample_scores.tsv"):
    key = (r["gse"], r["comparison"])
    if key not in order:
        order.append(key)
    scores[key][r["group"]].append(float(r["score"]))
auc = [r for r in rows_of("auc.tsv") if r["reproduced_auc"]]

pf.use_style()
ncols = 4
nrows = (len(order) + ncols - 1) // ncols
fig, axes = pf.figure(pf.TWO_COLUMN_MM, 42 * nrows, ncols=ncols, nrows=nrows, squeeze=False)
flat = [ax for row in axes for ax in row]
for ax, key, row in zip(flat, order, auc, strict=False):
    groups = scores[key]
    parts = ax.violinplot([groups["pre"], groups["post"]], positions=[0, 1], showextrema=False, widths=0.8)
    for body in parts["bodies"]:
        body.set_facecolor("#dddddd")
        body.set_alpha(1)
    for x, name, colour in ((0, "pre", pf.OKABE_ITO["blue"]), (1, "post", pf.OKABE_ITO["vermillion"])):
        ax.plot([x] * len(groups[name]), groups[name], ".", color=colour, label=f"{name}-exercise group")
    ax.set_xticks([0, 1], [f"n = {row['n_pre']}", f"n = {row['n_post']}"])
    ax.set_xlim(-0.6, 1.6)
    ax.set_title(
        f"{row['GSE']} {key[1].replace('pre_vs_', '').replace('main', '')}\npublished {row['published_auc']}, this test {row['reproduced_auc']}"
    )
axes[0][0].set_ylabel("Score, this test's method")
for ax in flat[len(order) :]:
    ax.set_visible(False)
pf.shared_legend(fig)
pf.save(fig, folder / "fig4_panels")
