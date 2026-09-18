#!/usr/bin/env python3
"""
generate_report.py

Write the markdown summary of the DisProt benchmark: the sweep's best ksize per
alphabet, then the headline tools per disorder stratum.

Usage:
    generate_report.py <all_disprot_metrics.parquet> <figures_dir> <output.md> [--headline-n 3]
"""

import argparse
import sys
from datetime import date
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from disprot_tool_names import alphabet_classes, annotate, headline_tools  # noqa: E402

MYA = {
    "mouse": 100, "chicken": 300, "zebrafish": 430,
    "ciona": 550, "fly": 600, "worm": 650,
    "yeast": 900, "arabidopsis": 1500, "ecoli": 2000,
}

DISORDER_CATEGORIES = ["ordered", "partial", "disordered"]
CAT_LABEL = {
    "ordered": "Ordered (mean disorder < 0.2)",
    "partial": "Partly disordered (0.2 to 0.5)",
    "disordered": "Disordered (> 0.5)",
}


def fmt(v) -> str:
    if v is None or v != v:
        return "-"
    return f"{v:.3f}"


def main(metrics_parquet: str, figures_dir: str, output_md: str, headline_n: int) -> None:
    metrics = pl.read_parquet(metrics_parquet)
    metrics = metrics.with_columns(
        [pl.col(c).fill_nan(None) for c in metrics.columns if metrics[c].dtype in (pl.Float32, pl.Float64)]
    )
    if "alphabet" not in metrics.columns:
        metrics = annotate(metrics)
    fig_path = Path(figures_dir)
    today = date.today().isoformat()

    all_metrics = metrics.filter(pl.col("disorder_category") == "all")
    n_pairs = all_metrics.select(pl.col("n_positives").max()).item()
    n_species = metrics["species"].n_unique()
    n_combos = metrics.filter(pl.col("alphabet").is_not_null())["tool"].n_unique()
    n_alphabets = metrics.filter(pl.col("alphabet").is_not_null())["alphabet"].n_unique()

    strat = (
        metrics.filter(pl.col("disorder_category").is_in(DISORDER_CATEGORIES))
        .group_by(["tool", "disorder_category"])
        .agg(
            pl.col("recall_at_fdr05").mean().alias("mean_recall"),
            pl.col("auc_pr").mean().alias("mean_auc_pr"),
        )
    )
    headline = headline_tools(metrics, headline_n)

    lines = [
        "# DisProt benchmark report",
        f"\nGenerated: {today}\n",
        "## What was run\n",
        "- Queries: human DisProt proteins that share at least one Pfam domain with a protein in "
        "at least three of the nine QfO target species.",
        f"- Targets: {n_species} QfO species proteomes, mouse (100 Mya) to E. coli (2000 Mya).",
        "- Truth: the Pfam QfO pair truth (a human-target pair is positive when the two proteins "
        "share a Pfam domain), filtered to the DisProt queries.",
        "- Strata: each query's mean metapredict disorder score, binned at 0.2 and 0.5.",
        f"- kmerseek: {n_combos} alphabet x ksize combo{'s' if n_combos != 1 else ''} over "
        f"{n_alphabets} alphabet{'s' if n_alphabets != 1 else ''}, the same "
        "matrix as the QfO region benchmark and the invertebrate dark set. A pair's score is the "
        "best max_containment over its regions.",
        "- Baseline: MMseqs2 at -s 7 on the same queries and targets.",
        f"- Positive pairs per species: up to {n_pairs}.\n",
    ]

    # -- The sweep --
    lines.append("## kmerseek sweep: each alphabet at its best k\n")
    lines.append("Ranked by mean recall at 5% FDR on the disordered queries, averaged over the "
                 "target species. The other columns read the same combo off the other strata.\n")
    km = strat.join(
        metrics.select(["tool", "alphabet", "ksize", "lowcomp"]).unique(), on="tool", how="left"
    ).filter(pl.col("alphabet").is_not_null())
    if len(km):
        best = (
            km.filter(pl.col("disorder_category") == "disordered")
            .sort("mean_recall", descending=True, nulls_last=True)
            .group_by(["alphabet", "lowcomp"], maintain_order=True).first()
            .sort("mean_recall", descending=True, nulls_last=True)
        )
        other = {(t, c): r for t, c, r in km.select(["tool", "disorder_category", "mean_recall"]).iter_rows()}
        masks = km["lowcomp"].unique().to_list()
        mask_col = len(masks) > 1
        head = "| Alphabet (classes) | " + ("Low-complexity removed | " if mask_col else "") + \
               "Best k | Recall, disordered | Recall, partial | Recall, ordered | AUC-PR, disordered |"
        lines.append(head)
        lines.append("|---|" + "---|" * (head.count("|") - 2))
        for row in best.iter_rows(named=True):
            t = row["tool"]
            cells = [f"{row['alphabet']} ({alphabet_classes(row['alphabet'])})"]
            if mask_col:
                cells.append(str(row["lowcomp"]).lower())
            cells += [str(row["ksize"]), fmt(row["mean_recall"]), fmt(other.get((t, "partial"))),
                      fmt(other.get((t, "ordered"))), fmt(row["mean_auc_pr"])]
            lines.append("| " + " | ".join(cells) + " |")
    else:
        lines.append("_No kmerseek combos in this run._")

    # -- Headline tools --
    lines.append(f"\n## Recall at 5% FDR by disorder stratum, headline tools\n")
    lines.append(f"The baselines and the {headline_n} kmerseek combos with the best mean recall on "
                 "the disordered queries. Mean across the target species.\n")
    lines.append("| Disorder stratum | " + " | ".join(headline) + " |")
    lines.append("|---|" + "---|" * len(headline))
    for cat in DISORDER_CATEGORIES:
        vals = []
        for tool in headline:
            v = strat.filter((pl.col("disorder_category") == cat) & (pl.col("tool") == tool))["mean_recall"].to_list()
            vals.append(fmt(v[0]) if v else "-")
        lines.append(f"| {CAT_LABEL[cat]} | " + " | ".join(vals) + " |")

    lines.append("\n## AUC-PR, all queries, mean across species\n")
    mean_auc = (
        all_metrics.filter(pl.col("tool").is_in(headline))
        .group_by("tool").agg(pl.col("auc_pr").mean().alias("mean_auc_pr"))
        .sort("mean_auc_pr", descending=True, nulls_last=True)
    )
    lines.append("| Tool | Mean AUC-PR |")
    lines.append("|---|---|")
    for row in mean_auc.iter_rows(named=True):
        lines.append(f"| {row['tool']} | {fmt(row['mean_auc_pr'])} |")

    lines.append("\n## Figures\n")
    figure_files = sorted(fig_path.glob("*.pdf"))
    if figure_files:
        lines += [f"- [{f.name}]({fig_path}/{f.name})" for f in figure_files]
    else:
        lines.append("_Figures not yet generated._")

    lines.append("\n## How to read the strata\n")
    lines.append("- Ordered: mean metapredict score below 0.2. Structured proteins; a "
                 "structure-based tool has something to compare here.")
    lines.append("- Partly disordered: 0.2 to 0.5. Structured domains with disordered loops or tails.")
    lines.append("- Disordered: above 0.5. Mostly without stable structure; the stratum this "
                 "benchmark exists for. A sequence k-mer method has no reason to drop here, and "
                 "a structure-based one has nothing to encode.")

    with open(output_md, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"Wrote {output_md}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("metrics_parquet")
    ap.add_argument("figures_dir")
    ap.add_argument("output_md")
    ap.add_argument("--headline-n", type=int, default=3)
    a = ap.parse_args()
    main(a.metrics_parquet, a.figures_dir, a.output_md, a.headline_n)
