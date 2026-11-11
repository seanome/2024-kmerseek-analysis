"""Draw the cross-series AUCs with the earlier values next to SCAN values, and apply scan_plan.md's rule on the gap.

Usage: plot_scan_cross.py <results/fig4 folder> <path to the repository's notebooks folder>
Earlier values: raw probe intensities for probe k-mers (cross_series/), the authors' RMA for HP k-mers
(hp_cross_series/). SCAN values: scan/cross_series/ and scan/hp_cross_series/; SCAN values without the probes SCAN.UPC never standardises
(the check added after the run in scan_plan.md): scan/without_unfitted/. One panel per kind of values, one block per
direction, one row per kind of k-mer. Grey dots are AUCs with pre and post swapped within each
person, orange dots AUCs of random k-mers with the chosen set's up count; the black diamond and coloured circles are the
1000 chosen k-mers. Writes scan/scan_cross.png and .svg, scan/scan_cross_summary.tsv and scan/scan_gap.tsv.
"""

import csv
import sys

import numpy as np

results, notebooks = sys.argv[1:]
sys.path.insert(0, notebooks)
import pubfig as pf  # noqa: E402

HEADLINE = "1000"
DIRECTIONS = [
    ("GSE28998_to_GSE27536", "Chosen on GSE28998,\nscored on GSE27536"),
    ("GSE27536_to_GSE28998", "Chosen on GSE27536,\nscored on GSE28998"),
]
KINDS = {
    "probe": ("D", "black", "25-letter probe k-mers"),
    "hp_lehninger2": ("o", pf.OKABE_ITO["blue"], "HP k-mers, hp_lehninger2 (hydrophobic AFGILMPVWY)"),
    "hp_thomas_dill2": ("o", pf.OKABE_ITO["vermillion"], "HP k-mers, hp_thomas_dill2 (hydrophobic ACFILMVWY)"),
}
VALUES = {
    "earlier": ("", "Earlier values: raw intensities\n(probe k-mers), authors' RMA (HP)"),
    "SCAN": ("scan/", "SCAN values, one array\nat a time, for both"),
    "SCAN without unstandardised": (
        "scan/without_unfitted/",
        "SCAN values without the 4999\nprobes per array left at 0",
    ),
}


def rows_of(path):
    with open(path) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def stem(values, direction, kind):
    folder = VALUES[values][0]
    if kind == "probe":
        return f"{results}/{folder}cross_series/{direction}"
    return f"{results}/{folder}hp_cross_series/{direction}.{kind}"


def result(values, direction, kind):
    path = stem(values, direction, kind)
    row = next(r for r in rows_of(f"{path}.cross.tsv") if r["kmers"] == HEADLINE)
    chance = {}
    for r in rows_of(f"{path}.cross_chance.tsv"):
        if r["kmers"] == HEADLINE:
            chance.setdefault(r["chance"], []).append(float(r["auc"]))
    return row, chance


summary = []
random_aucs = {}
pf.use_style()
fig, axes = pf.figure(pf.TWO_COLUMN_MM, 95, ncols=3, sharey=True)
ax0 = axes[0]
ax0.plot([], [], ".", color=pf.GREY, label="Pre and post swapped within each person")
ax0.plot([], [], ".", color=pf.OKABE_ITO["orange"], label="Random k-mers, same up count")
for marker, colour, label in KINDS.values():
    ax0.plot([], [], marker, color=colour, label=f"{label}, 1000 chosen")
rng = np.random.default_rng(0)
for ax, (values, (_, title)) in zip(axes, VALUES.items(), strict=True):
    ticks, labels = [], []
    y = 0.0
    for direction, label in DIRECTIONS[::-1]:
        block = []
        for kind, (marker, colour, _) in KINDS.items():
            row, chance = result(values, direction, kind)
            for chance_kind, aucs in chance.items():
                colour_of = pf.GREY if chance_kind == "labels swapped" else pf.OKABE_ITO["orange"]
                offset = -0.12 if chance_kind == "labels swapped" else 0.12
                ax.plot(
                    aucs,
                    y + offset + rng.uniform(-0.05, 0.05, len(aucs)),
                    ".",
                    color=colour_of,
                    markersize=2,
                    alpha=0.5,
                )
            random_aucs[values, direction, kind] = np.array(chance["random k-mers"])
            auc = float(row["auc"])
            ax.plot(auc, y, marker, color=colour, zorder=5)
            p_swapped, p_random = float(row["p_labels_swapped"]), float(row["p_random_kmers"])
            ax.text(1.04, y, f"{auc:.2f}\n({p_swapped:.3f}, {p_random:.3f})", va="center", ha="left")
            summary.append(
                {
                    "test": direction,
                    "kmers": kind,
                    "values": values,
                    "auc": row["auc"],
                    "p_labels_swapped": row["p_labels_swapped"],
                    "p_random_kmers": row["p_random_kmers"],
                    "median_auc_random_kmers": row["median_auc_random_kmers"],
                    "people_post_above_pre": f"{row['people_post_above_pre']} of {row['people']}",
                    "people": row["people"],
                    "kmers_up_after_in_training": row["kmers_up_after_in_training"],
                    "probe_sets_behind": row.get("probe_sets_behind", ""),
                    "carries_over": p_swapped < 0.05 and p_random < 0.05,
                }
            )
            block.append(y)
            y += 0.45
        ticks.append(np.mean(block))
        labels.append(label)
        y += 0.5
    ax.set_yticks(ticks, labels)
    ax.set_ylim(-0.4, y - 0.5)
    ax.set_xlim(0, 1.5)
    ax.spines["bottom"].set_bounds(0, 1)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.axvline(0.5, color=pf.GREY, linestyle="--", linewidth=0.5, zorder=1)
    ax.set_xlabel("AUC, 1000 chosen k-mers\n(text: AUC; p against swapped\nlabels, against random k-mers)")
    ax.set_title(title, loc="left")
    pf.panel_label(ax, "abc"[list(VALUES).index(values)])
pf.shared_legend(fig, ncol=2)
pf.save(fig, f"{results}/scan/scan_cross")


def write(path, rows_out):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_out[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows_out)


write(f"{results}/scan/scan_cross_summary.tsv", summary)

# scan_plan.md: chosen on GSE27536 and scored on GSE28998, is the random HP median still well above the probe one?
# Each median is over 200 random k-mer sets, so the difference has a spread: resample each set of 200 AUCs with
# replacement (10000 times) and take the middle 95 % of the differences (added after the analysis review of 0eca190d).
# The scored series has the same people before and after, so its AUC moves in steps of 1 / people^2.
scored = DIRECTIONS[1][0]
median = {(r["values"], r["kmers"]): float(r["median_auc_random_kmers"]) for r in summary if r["test"] == scored}
people = {int(r["people"]) for r in summary if r["test"] == scored}
assert len(people) == 1, people
step = 1 / people.pop() ** 2
boot = np.random.default_rng(1)


def resampled_medians(aucs, draws=10_000):
    return np.median(boot.choice(aucs, (draws, len(aucs))), axis=1)


def reading_of(difference):
    if abs(difference) <= 0.05:
        return "the values explain the gap"
    return "the values do not explain the gap" if difference >= 0.1 else "unclear"


def readings_within(low, high):
    """Every reading that some difference between low and high would get, by reading_of's cut-offs (0.05 and 0.1)."""
    covered = []
    if low <= 0.05 and high >= -0.05:
        covered.append("the values explain the gap")
    if low < -0.05 or (low < 0.1 and high > 0.05):
        covered.append("unclear")
    if high >= 0.1:
        covered.append("the values do not explain the gap")
    return covered


gap = []
for values in VALUES:
    probe = resampled_medians(random_aucs[values, scored, "probe"])
    for kind in ("hp_lehninger2", "hp_thomas_dill2"):
        difference = median[values, kind] - median[values, "probe"]
        low, high = np.percentile(resampled_medians(random_aucs[values, scored, kind]) - probe, [2.5, 97.5])
        gap.append(
            {
                "values": values,
                "alphabet": kind,
                "random_hp_median_auc": f"{median[values, kind]:.3f}",
                "random_probe_median_auc": f"{median[values, 'probe']:.3f}",
                "difference": f"{difference:.3f}",
                "difference_95pct_low": f"{low:.3f}",
                "difference_95pct_high": f"{high:.3f}",
                "auc_step": f"{step:.3f}",
                "reading": reading_of(difference) if values != "earlier" else "",
                "readings_within_95pct": " / ".join(readings_within(low, high)) if values != "earlier" else "",
            }
        )
write(f"{results}/scan/scan_gap.tsv", gap)
for line in summary + gap:
    print("\t".join(map(str, line.values())))
