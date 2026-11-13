"""Draw the held-out AUC of probe k-mers next to the published gene-signature AUC, and each person's change.

Usage: plot_held_out.py <results/fig4 folder> <path to the repository's notebooks folder>
Panel a: one row per comparison. Grey dots are the AUCs after swapping pre and post labels within each person;
the grey band is within 0.15 of the published AUC, turned so that 1 means post above pre. Panel b: each person's
mean held-out score after minus before, from the 1000 k-mers chosen without that person. Writes held_out.png and
.svg to held_out.
"""

import csv
import sys
from collections import defaultdict

import numpy as np

results, notebooks = sys.argv[1:]
sys.path.insert(0, notebooks)
import pubfig as pf  # noqa: E402

RULE_DISTANCE = 0.15
HEADLINE = "1000"
# Published AUCs from Domen 2026 Fig 4, turned so that 1 means post above pre; GSE111555 is PR 212's 0.809
RUNS = [
    ("GSE83578", "pre_vs_30min", "GSE83578, 30 min after", 1 - 0.07),
    ("GSE28998", "main", "GSE28998, 12 weeks training", 0.80),
    ("GSE111555", "main", "GSE111555, 18 weeks training", 0.809),
]


def rows_of(path):
    with open(path) as f:
        return list(csv.DictReader(f, delimiter="\t"))


folder = f"{results}/held_out"
pf.use_style()
fig, (ax, change) = pf.figure(pf.TWO_COLUMN_MM, 62, ncols=2, width_ratios=[3, 2])
y = np.arange(len(RUNS))[::-1]
rng = np.random.default_rng(0)
ax.axvline(0.5, color=pf.GREY, linestyle="--", label="0.5 = no difference", zorder=1)
ax.plot([], [], "s", color="#D9D9D9", markersize=6, label="Within 0.15 of published")
ax.plot([], [], ".", color=pf.GREY, label="Labels swapped within each person")
ax.plot([], [], "D", color="black", label="Published gene signature")
ax.plot([], [], "s", mfc="white", mec=pf.OKABE_ITO["blue"], label="k-mers chosen with everyone")
ax.plot([], [], "o", color=pf.OKABE_ITO["blue"], label="k-mers chosen without the held-out person")
for row_y, (gse, comparison, name, published) in zip(y, RUNS, strict=True):
    summary = next(r for r in rows_of(f"{folder}/{gse}.{comparison}.held_out.tsv") if r["kmers"] == HEADLINE)
    swaps = [float(r["auc"]) for r in rows_of(f"{folder}/{gse}.{comparison}.held_out_shuffles.tsv") if r["kmers"] == HEADLINE]
    low, high = published - RULE_DISTANCE, min(1, published + RULE_DISTANCE)
    ax.barh(row_y, high - low, left=low, height=0.6, color="#D9D9D9", zorder=0)
    ax.plot(swaps, row_y + rng.uniform(-0.2, 0.2, len(swaps)), ".", color=pf.GREY, markersize=2, alpha=0.6, zorder=2)
    ax.plot(published, row_y, "D", color="black", zorder=3)
    ax.plot(float(summary["auc_kmers_chosen_from_everyone"]), row_y + 0.25, "s", mfc="white", mec=pf.OKABE_ITO["blue"], zorder=4)
    ax.plot(float(summary["held_out_auc"]), row_y, "o", color=pf.OKABE_ITO["blue"], zorder=5)
    ax.text(1.06, row_y, f"{float(summary['held_out_auc']):.2f}\np {float(summary['p']):.3f}", va="center", ha="left")
ax.set_yticks(y, [name for _, _, name, _ in RUNS])
ax.set_ylim(-0.6, len(RUNS) - 0.4)
ax.set_xlim(0, 1.04)
ax.set_xticks(np.linspace(0, 1, 6))
ax.spines["bottom"].set_bounds(0, 1)
ax.set_xlabel("AUC: chance a post sample scores above a pre sample")

for row_y, (gse, comparison, name, _) in zip(y, RUNS, strict=True):
    scores = defaultdict(lambda: defaultdict(list))
    for r in rows_of(f"{folder}/{gse}.{comparison}.held_out_samples.tsv"):
        scores[r["person"]][r["group"]].append(float(r[f"score_{HEADLINE}"]))
    differences = [np.mean(s["post"]) - np.mean(s["pre"]) for s in scores.values()]
    change.plot(differences, row_y + rng.uniform(-0.15, 0.15, len(differences)), "o", color=pf.OKABE_ITO["blue"], markersize=3)
    above = sum(d > 0 for d in differences)
    change.text(max(differences) + 0.1, row_y, f"{above} of {len(differences)} up", va="center", ha="left")
change.axvline(0, color=pf.GREY, linestyle="--")
change.set_yticks(y, [])
change.set_ylim(-0.6, len(RUNS) - 0.4)
change.set_xlim(min(0, change.get_xlim()[0]), change.get_xlim()[1] + 0.6)
change.set_xlabel("Held-out score after minus before,\none dot per person (s.d. units)")
pf.shared_legend(fig, ncol=3)
for a, letter in zip((ax, change), "ab"):
    pf.panel_label(a, letter)
pf.save(fig, f"{folder}/held_out")
