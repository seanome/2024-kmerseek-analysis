"""Draw how many k-mers and SPLASH anchors differ between pre and post, against shuffled groups.

Usage: plot_splash_kamrat.py <splash_kamrat results folder> <published_fig4.tsv> <proteins.fasta> <notebooks folder>
Panel a: KaMRaT's k-mers below 5 % false discovery. Panel b: SPLASH anchors whose target use differs between
groups. Grey marks are the shuffled groupings. A comparison counts as mixed when pre and post samples share
scan dates (scan_dates.tsv) or, for GSE104079, array slides; otherwise a group difference cannot be told from
a batch difference. Writes summary.tsv, summary.png and summary.svg to the folder.
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

folder, published_file, proteins, notebooks = sys.argv[1:]
sys.path.insert(0, notebooks)
import pubfig as pf  # noqa: E402
from geo2kmers.landing import land_probes  # noqa: E402

ORDER = [
    ("GSE83578", "pre_vs_30min", "GSE83578, 30 min after", True),
    ("GSE83578", "pre_vs_24h", "GSE83578, 24 h after", False),
    ("GSE28998", "main", "GSE28998, 12 weeks training", True),
    ("GSE111555", "main", "GSE111555, 18 weeks training", True),
    ("GSE83578", "pre_vs_3h", "GSE83578, 3 h after", False),
    ("GSE27536", "main", "GSE27536, after exercise", False),
    ("GSE104079", "main", "GSE104079, 45 min after", False),
]


def rows_of(path):
    with open(path) as f:
        return list(csv.DictReader(f, delimiter="\t"))


published = {(r["dataset"], r["comparison"]): r["auc"] for r in rows_of(published_file) if r["panel"] in ("4C", "4D")}
kmers = defaultdict(dict)
for path in Path(folder, "kamrat").glob("*.kamrat_summary.tsv"):
    r = rows_of(path)[0]
    kmers[r["dataset"], r["comparison"]][int(r["seed"])] = (int(r["kmers_tested"]), int(r["kmers_below_fdr"]))
anchors = {}
for path in Path(folder, "splash").glob("*.splash_summary.tsv"):
    for r in rows_of(path):
        anchors[r["dataset"], r["comparison"]] = r


def on_signature(dataset, comparison):
    """How many of the 1000 best k-mers land on a signature protein, read as given or complemented."""
    path = Path(folder, "kamrat", f"{dataset}.{comparison}.0.kamrat_top.tsv")
    top = {r["kmer"]: r["kmer"] for r in rows_of(path) if float(r["q"]) < 0.05}
    complement = str.maketrans("ACGT", "TGCA")
    if not top:
        return 0
    landed = {row[0] for row in land_probes(top, [proteins])[2]}
    landed |= {row[0] for row in land_probes({k: k.translate(complement) for k in top}, [proteins])[2]}
    return len(landed)


table = []
for dataset, comparison, name, mixed in ORDER:
    runs = kmers[dataset, comparison]
    shuffled = [v[1] for seed, v in runs.items() if seed]
    splash = anchors[dataset, comparison]
    table.append(
        {
            "comparison": name,
            "pre_and_post_share_batches": mixed,
            "published_auc": published[dataset, comparison],
            "kmers_tested": runs[0][0],
            "kmers_below_fdr": runs[0][1],
            "highest_kmers_below_fdr_shuffled": max(shuffled),
            "best_kmers_on_signature_proteins": on_signature(dataset, comparison),
            "splash_significant_anchors": splash["significant_anchors"],
            "splash_anchors_inside_one_feature": splash["anchors_inside_one_feature"],
            "splash_anchors_differing_between_groups": splash["anchors_differing_between_groups"],
            "highest_splash_anchors_shuffled": splash["highest_with_shuffled_groups"],
            "splash_hits_inside_one_feature": splash["hits_inside_one_feature"],
            "splash_hits_on_signature_proteins": splash["hits_on_signature_proteins"],
        }
    )
with open(f"{folder}/summary.tsv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(table[0]), delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerows(table)

pf.use_style()
fig, (ax_a, ax_b) = pf.figure(pf.TWO_COLUMN_MM, 76, ncols=2, sharey=True)
y = list(range(len(table)))[::-1]
panels = (
    (ax_a, "kmers_below_fdr", "highest_kmers_below_fdr_shuffled", "KaMRaT: k-mers differing between groups"),
    (
        ax_b,
        "splash_anchors_differing_between_groups",
        "highest_splash_anchors_shuffled",
        "SPLASH: anchors differing between groups",
    ),
)
for ax, real, shuffled, label in panels:
    # Zero has no place on a ratio axis, so a count of 0 sits at 0.5, the tick labelled 0
    place = lambda value: max(float(value), 0.5)
    for row_y, row in zip(y, table, strict=True):
        ax.plot([place(row[shuffled]), place(row[real])], [row_y, row_y], color="#D9D9D9", zorder=1)
    ax.plot([place(r[shuffled]) for r in table], y, "X", color=pf.GREY, label="Highest with shuffled groups", zorder=2)
    for mixed, marker, colour, text in (
        (True, "o", pf.OKABE_ITO["blue"], "True groups, pre and post share batches"),
        (False, "s", pf.OKABE_ITO["vermillion"], "True groups, pre and post mostly in separate batches"),
    ):
        chosen = [
            (place(r[real]), row_y)
            for r, row_y in zip(table, y, strict=True)
            if r["pre_and_post_share_batches"] == mixed
        ]
        ax.plot(*zip(*chosen, strict=True), marker, color=colour, label=text, zorder=3)
    ax.set_xscale("log")
    ticks = [0.5, 1, 10, 100, 1000, 10000, 100000, 1000000]
    ax.set_xticks(ticks, ["0", "1", "10", "100", "1000", "10000", "100000", "1000000"])
    ax.set_xlim(0.35, 2000000)
    ax.minorticks_off()
    ax.set_xlabel(label)
ax_a.set_yticks(y, [f"{r['comparison']} (AUC {float(r['published_auc']):.2f})" for r in table])
pf.shared_legend(fig)
for ax, letter in zip((ax_a, ax_b), "ab", strict=True):
    pf.panel_label(ax, letter)
pf.save(fig, f"{folder}/summary")
