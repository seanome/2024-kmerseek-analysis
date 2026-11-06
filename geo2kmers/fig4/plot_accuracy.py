"""Draw how close geo2kmers comes to each published AUC, on the corrected GSE111555 groups.

Usage: plot_accuracy.py <results/fig4 folder> <path to the repository's notebooks folder>
One row per comparison, from most down to most up after exercise. The grey band is the rule's range of
0.15 around the published AUC, and the columns give each score's distance from it. GSE111555's published
AUC is PR 212's 0.809 on the corrected groups, not a printed number. Writes accuracy.tsv, .png and .svg to
corrected_labels.
"""

import csv
import sys

import numpy as np

results, notebooks = sys.argv[1:]
sys.path.insert(0, notebooks)
import pubfig as pf  # noqa: E402

RULE_DISTANCE = 0.15
SCORES = [
    ("Planned score", "reproduced_auc", "planned", "o", pf.OKABE_ITO["blue"]),
    ("Second score", "auc_ghosh_weights", "ghosh_weights", "s", pf.OKABE_ITO["vermillion"]),
]
NAMES = {
    ("GSE83578", "Pre vs Post 30M"): "GSE83578, 30 min after",
    ("GSE83578", "Pre vs Post 3H"): "GSE83578, 3 h after",
    ("GSE83578", "Pre vs Post 24H"): "GSE83578, 24 h after",
    ("GSE27536", "Pre vs Post"): "GSE27536, after exercise",
    ("GSE104079", "Pre vs 45 Min Post"): "GSE104079, 45 min after",
    ("GSE28998", "Baseline (Sed) vs Post 12 Weeks (Ex)"): "GSE28998, 12 weeks training",
    ("GSE111555", "Baseline (Sed) vs Post 18 Weeks (Ex)"): "GSE111555, 18 weeks training",
}


def rows_of(path):
    with open(path) as f:
        return list(csv.DictReader(f, delimiter="\t"))


folder = f"{results}/corrected_labels"
rows = [r for r in rows_of(f"{folder}/auc.tsv") if r["reproduced_auc"]]
rows.sort(key=lambda r: (float(r["published_auc"]), r["group_pair"]), reverse=True)
chance = {
    r["score"]: float(r["mean_passing_by_chance"])
    for r in rows_of(f"{folder}/chance_test.tsv")
    if r["model"] == "shuffled labels"
}

table = []
for r in rows:
    published = float(r["published_auc"])
    table.append({"comparison": NAMES[r["GSE"], r["group_pair"]], "published_auc": published})
    for _, column, score, _, _ in SCORES:
        table[-1][f"{score}_auc"] = float(r[column])
        table[-1][f"{score}_distance"] = round(abs(float(r[column]) - published), 3)
with open(f"{folder}/accuracy.tsv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(table[0]), delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerows(table)

pf.use_style()
fig, (ax, numbers) = pf.figure(pf.TWO_COLUMN_MM, 80, ncols=2, width_ratios=[2, 1], sharey=True)
y = np.arange(len(table))[::-1]
ax.axvline(0.5, color=pf.GREY, linestyle="--", label="0.5 = no difference between groups", zorder=1)
for row_y, row in zip(y, table, strict=True):
    published = row["published_auc"]
    low, high = max(0, published - RULE_DISTANCE), min(1, published + RULE_DISTANCE)
    ax.barh(row_y, high - low, left=low, height=0.6, color="#D9D9D9", zorder=0)
ax.plot([], [], "s", color="#D9D9D9", markersize=6, label="Within 0.15 of the published AUC")
ax.plot([r["published_auc"] for r in table], y, "D", color="black", label="Published AUC", zorder=3)
for label, _, score, marker, colour in SCORES:
    # Each score keeps its own line in the row, so equal AUCs do not hide each other
    line = y + (0.13 if score == "planned" else -0.13)
    ax.plot([r[f"{score}_auc"] for r in table], line, marker, color=colour, label=f"{label}, geo2kmers", zorder=4)
    x = 0.25 if score == "planned" else 0.75
    numbers.text(x, len(table) - 0.45, label + "\ndistance", ha="center", va="bottom")
    for row_y, row in zip(y, table, strict=True):
        distance = row[f"{score}_distance"]
        mark = "yes" if distance <= RULE_DISTANCE else "no"
        numbers.text(x, row_y, f"{distance:.2f} ({mark})", ha="center", va="center")
    passing = sum(r[f"{score}_distance"] <= RULE_DISTANCE for r in table)
    mean_distance = np.mean([r[f"{score}_distance"] for r in table])
    summary = f"{passing} of {len(table)} within 0.15\n{chance[score]:.1f} by chance\nmean distance {mean_distance:.2f}"
    numbers.text(x, -1.0, summary, ha="center", va="top")
ax.set_yticks(y, [r["comparison"] for r in table])
ax.set_ylim(-0.6, len(table) - 0.4)
numbers.set_xlim(0, 1)
numbers.axis("off")
ax.set_xlim(0, 1)
ax.set_xlabel("AUC: chance a post sample scores above a pre sample")
pf.shared_legend(fig)
pf.save(fig, f"{folder}/accuracy")
