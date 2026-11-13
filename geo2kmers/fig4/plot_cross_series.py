"""Draw the AUC of k-mers chosen on one U133 Plus 2 series and scored on the other, against both kinds of chance.

Usage: plot_cross_series.py <results/fig4 folder> <path to the repository's notebooks folder>
Panel a: one row per direction. Grey dots are AUCs with pre and post swapped within each person of the scored series;
orange dots are AUCs of random k-mers with the same up and down split; the blue dot is the AUC of the 1000 chosen
k-mers. Panel b: each scored person's score after minus before. Writes cross_series.png and .svg to cross_series.
"""

import csv
import sys
from collections import defaultdict

import numpy as np

results, notebooks = sys.argv[1:]
sys.path.insert(0, notebooks)
import pubfig as pf  # noqa: E402

HEADLINE = "1000"
RUNS = [
    ("GSE28998", "GSE27536", "Chosen on GSE28998 (biceps, 7),\nscored on GSE27536 (thigh, 27)"),
    ("GSE27536", "GSE28998", "Chosen on GSE27536,\nscored on GSE28998"),
]
CHANCE = {
    "labels swapped": (pf.GREY, -0.15, "Pre and post swapped within each person"),
    "random k-mers": (pf.OKABE_ITO["orange"], 0.15, "Random k-mers, same up and down split"),
}


def rows_of(path):
    with open(path) as f:
        return list(csv.DictReader(f, delimiter="\t"))


folder = f"{results}/cross_series"
pf.use_style()
fig, (ax, change) = pf.figure(pf.TWO_COLUMN_MM, 55, ncols=2, width_ratios=[3, 2])
y = np.arange(len(RUNS))[::-1]
rng = np.random.default_rng(0)
ax.axvline(0.5, color=pf.GREY, linestyle="--", label="0.5 = no difference", zorder=1)
for colour, _, label in CHANCE.values():
    ax.plot([], [], ".", color=colour, label=label)
ax.plot([], [], "o", color=pf.OKABE_ITO["blue"], label="1000 chosen k-mers")
for row_y, (train, scored, _) in zip(y, RUNS, strict=True):
    prefix = f"{folder}/{train}_to_{scored}"
    summary = next(r for r in rows_of(f"{prefix}.cross.tsv") if r["kmers"] == HEADLINE)
    chance = [r for r in rows_of(f"{prefix}.cross_chance.tsv") if r["kmers"] == HEADLINE]
    for kind, (colour, offset, _) in CHANCE.items():
        aucs = [float(r["auc"]) for r in chance if r["chance"] == kind]
        ax.plot(aucs, row_y + offset + rng.uniform(-0.08, 0.08, len(aucs)), ".", color=colour, markersize=2, alpha=0.6)
    auc = float(summary["auc"])
    ax.plot(auc, row_y, "o", color=pf.OKABE_ITO["blue"], zorder=5)
    ax.text(
        1.06,
        row_y,
        f"{auc:.2f}\np {float(summary['p_labels_swapped']):.3f} swapped\np {float(summary['p_random_kmers']):.3f} random",
        va="center",
        ha="left",
    )
ax.set_yticks(y, [name for *_, name in RUNS])
ax.set_ylim(-0.6, len(RUNS) - 0.4)
ax.set_xlim(0, 1.04)
ax.set_xticks(np.linspace(0, 1, 6))
ax.spines["bottom"].set_bounds(0, 1)
ax.set_xlabel("AUC in the scored series: chance a post\nsample scores above a pre sample")

for row_y, (train, scored, _) in zip(y, RUNS, strict=True):
    scores = defaultdict(lambda: defaultdict(list))
    for r in rows_of(f"{folder}/{train}_to_{scored}.cross_samples.tsv"):
        scores[r["person"]][r["group"]].append(float(r[f"score_{HEADLINE}"]))
    differences = [np.mean(s["post"]) - np.mean(s["pre"]) for s in scores.values() if s["post"] and s["pre"]]
    change.plot(
        differences, row_y + rng.uniform(-0.15, 0.15, len(differences)), "o", color=pf.OKABE_ITO["blue"], markersize=3
    )
    change.text(
        max(differences) + 0.05,
        row_y,
        f"{sum(d > 0 for d in differences)} of {len(differences)} up",
        va="center",
        ha="left",
    )
change.axvline(0, color=pf.GREY, linestyle="--")
change.set_yticks(y, [])
change.set_ylim(-0.6, len(RUNS) - 0.4)
change.set_xlim(min(0, change.get_xlim()[0]), change.get_xlim()[1] + 0.3)
change.set_xlabel("Score after minus before,\none dot per scored person (s.d. units)")
pf.shared_legend(fig, ncol=2)
for a, letter in zip((ax, change), "ab"):
    pf.panel_label(a, letter)
pf.save(fig, f"{folder}/cross_series")
