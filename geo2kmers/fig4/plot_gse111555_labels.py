"""Draw why six GSE111555 samples moved and what that did to the AUC.

Usage: plot_gse111555_labels.py <results/fig4 folder> <path to the repository's notebooks folder>
Panel a: each PBMC sample titled after training, as its VO2peak minus that man's baseline PBMC sample.
Panel b: the AUCs on the authors' groups and on the corrected groups, with the rule's range of 0.15.
"""

import csv
import re
import sys

import numpy as np

results, notebooks = sys.argv[1:]
sys.path.insert(0, notebooks)
import pubfig as pf  # noqa: E402

RULE_DISTANCE = 0.15
SCORES = [
    ("Published score, from PR 212", "published_auc", "D", "black"),
    ("Planned score, geo2kmers", "reproduced_auc", "o", pf.OKABE_ITO["blue"]),
    ("Second score, geo2kmers", "auc_ghosh_weights", "s", pf.OKABE_ITO["vermillion"]),
]


def rows_of(path):
    with open(path) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def man_of(title):
    return re.match(r"\d+", title).group()


def gse111555(path):
    return next(r for r in rows_of(path) if r["GSE"] == "GSE111555")


labels = rows_of(f"{results}/corrected_labels/gse111555_labels.tsv")
baseline = {man_of(r["title"]): float(r["vo2peak_l_min"]) for r in labels if r["by_title"] == "pre"}
change = {"post": [], "pre": []}
for r in labels:
    if r["by_title"] == "post":
        change[r["by_source"]].append(float(r["vo2peak_l_min"]) - baseline[man_of(r["title"])])

pf.use_style()
fig, (ax_a, ax_b) = pf.figure(pf.TWO_COLUMN_MM, 62, ncols=2)
rng = np.random.default_rng(1)
for x, values in enumerate(change.values()):
    ax_a.plot(x + rng.uniform(-0.12, 0.12, len(values)), values, "o", color=pf.GREY, markeredgecolor="black")
ax_a.axhline(0, color=pf.GREY, linestyle="--")
ax_a.annotate("No change from baseline", (0.5, 0.01), va="bottom", ha="center")
ax_a.set_xticks([0, 1], ["Source field:\nposttraining (10)", "Source field:\npretraining (6)"])
ax_a.set_xlim(-0.5, 1.5)
ax_a.set_ylabel("VO2peak change from baseline (L/min)")
ax_a.set_xlabel("PBMC samples titled posttraining")

groupings = [
    ("Authors' groups\n22 vs 10", gse111555(f"{results}/auc.tsv")),
    ("Corrected groups\n16 vs 16", gse111555(f"{results}/corrected_labels/auc.tsv")),
]
for y, (_, row) in enumerate(groupings):
    centre = float(row["published_auc"])
    ax_b.plot([centre - RULE_DISTANCE, min(centre + RULE_DISTANCE, 1)], [y, y], color=pf.GREY, linewidth=5, alpha=0.4)
for name, column, marker, colour in SCORES:
    values = [float(row[column]) for _, row in groupings]
    ax_b.plot(values, range(len(values)), marker, color=colour, linestyle="none", label=name)
ax_b.plot([], [], color=pf.GREY, linewidth=5, alpha=0.4, label="Within 0.15 of the published score")
ax_b.set_yticks(range(len(groupings)), [name for name, _ in groupings])
ax_b.set_ylim(-0.6, 1.6)
ax_b.set_xlim(0.6, 1)
ax_b.set_xlabel("AUC, post scoring above pre (0.5 = no difference)")
pf.shared_legend(fig)
for ax, letter in zip((ax_a, ax_b), "ab"):
    pf.panel_label(ax, letter)
pf.save(fig, f"{results}/corrected_labels/gse111555_labels")
