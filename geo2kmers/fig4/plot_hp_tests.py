"""Draw the held-out and cross-series AUCs of 30-letter HP k-mers next to those of the 25-letter probe k-mers.

Usage: plot_hp_tests.py <results/fig4 folder> <path to the repository's notebooks folder>
One block per test, one row per HP alphabet. Grey dots are AUCs with pre and post swapped within each person; orange
dots, for the cross-series tests, are AUCs of random k-mers with the same up and down split. The coloured circle is
the AUC of the 1000 chosen HP k-mers and the black diamond that of the 1000 probe k-mers. Writes hp_tests.png and .svg
to hp_tests.
"""

import csv
import sys
from pathlib import Path

import numpy as np

results, notebooks = sys.argv[1:]
sys.path.insert(0, notebooks)
import pubfig as pf  # noqa: E402

HEADLINE = "1000"
ALPHABETS = {
    "hp_lehninger2": (pf.OKABE_ITO["blue"], "HP k-mers, hp_lehninger2 (hydrophobic AFGILMPVWY)"),
    "hp_thomas_dill2": (pf.OKABE_ITO["vermillion"], "HP k-mers, hp_thomas_dill2 (hydrophobic ACFILMVWY)"),
}
# Kind, name in the files, label
TESTS = [
    ("held out", "GSE28998", "GSE28998 (biceps, 7 people),\neach person held out"),
    ("held out", "GSE111555", "GSE111555 (thigh, 16 people),\neach person held out"),
    ("cross", "GSE28998_to_GSE27536", "Chosen on GSE28998,\nscored on GSE27536 (27 people)"),
    ("cross", "GSE27536_to_GSE28998", "Chosen on GSE27536,\nscored on GSE28998 (7 people)"),
]


def rows_of(path):
    with open(path) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def headline(path):
    return next(r for r in rows_of(path) if r["kmers"] == HEADLINE)


def result(kind, name, alphabet):
    """The AUC, its p values as text, the chance AUCs by kind, and the probe sets behind the chosen k-mers."""
    if kind == "held out":
        stem = f"{results}/hp_held_out/{name}.main.{alphabet}"
        row = headline(f"{stem}.held_out.tsv")
        swapped = [float(r["auc"]) for r in rows_of(f"{stem}.held_out_shuffles.tsv") if r["kmers"] == HEADLINE]
        behind = f"{row['fewest_probe_sets_behind']}-{row['most_probe_sets_behind']}"
        return float(row["held_out_auc"]), f"p {float(row['p']):.3f}", {"labels swapped": swapped}, behind
    stem = f"{results}/hp_cross_series/{name}.{alphabet}"
    row = headline(f"{stem}.cross.tsv")
    chance = {}
    for r in rows_of(f"{stem}.cross_chance.tsv"):
        if r["kmers"] == HEADLINE:
            chance.setdefault(r["chance"], []).append(float(r["auc"]))
    text = f"p {float(row['p_labels_swapped']):.3f} swapped, {float(row['p_random_kmers']):.3f} random"
    return float(row["auc"]), text, chance, row["probe_sets_behind"]


def probe_kmer_auc(kind, name):
    if kind == "held out":
        return float(headline(f"{results}/held_out/{name}.main.held_out.tsv")["held_out_auc"])
    return float(headline(f"{results}/cross_series/{name}.cross.tsv")["auc"])


pf.use_style()
fig, ax = pf.figure(pf.TWO_COLUMN_MM, 95)
rng = np.random.default_rng(0)
ax.axvline(0.5, color=pf.GREY, linestyle="--", label="0.5 = no difference", zorder=1)
ax.plot([], [], ".", color=pf.GREY, label="Pre and post swapped within each person")
ax.plot([], [], ".", color=pf.OKABE_ITO["orange"], label="Random k-mers, same up and down split")
ax.plot([], [], "D", color="black", label="25-letter probe k-mers")
for colour, label in ALPHABETS.values():
    ax.plot([], [], "o", color=colour, label=label)
ticks, labels, table = [], [], []
y = 0.0
for kind, name, label in TESTS[::-1]:
    block = []
    for alphabet, (colour, _) in ALPHABETS.items():
        auc, text, chance, behind = result(kind, name, alphabet)
        for chance_kind, aucs in chance.items():
            colour_of = pf.GREY if chance_kind == "labels swapped" else pf.OKABE_ITO["orange"]
            offset = -0.12 if chance_kind == "labels swapped" else 0.12
            ax.plot(
                aucs, y + offset + rng.uniform(-0.06, 0.06, len(aucs)), ".", color=colour_of, markersize=2, alpha=0.5
            )
        ax.plot(auc, y, "o", color=colour, zorder=5)
        ax.text(1.05, y, f"{auc:.2f}, {text}; {behind} probe sets", va="center", ha="left")
        block.append(y)
        table.append((name, alphabet, f"{auc:.3f}", text, behind))
        y += 0.5
    ax.plot(probe_kmer_auc(kind, name), np.mean(block), "D", color="black", zorder=6)
    ticks.append(np.mean(block))
    labels.append(label)
    y += 0.6
ax.set_yticks(ticks, labels)
ax.set_ylim(-0.5, y - 0.6)
ax.set_xlim(0, 1.03)
ax.spines["bottom"].set_bounds(0, 1)
ax.set_xticks(np.linspace(0, 1, 6))
ax.set_xlabel("AUC: chance a post sample scores above a pre sample, with the 1000 chosen k-mers")
pf.shared_legend(fig, ncol=2)
out = Path(results) / "hp_tests"
out.mkdir(exist_ok=True)
pf.save(fig, f"{out}/hp_tests")
for line in table:
    print("\t".join(line))
