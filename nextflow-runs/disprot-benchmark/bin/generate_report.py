#!/Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3
"""
generate_report.py

Write a summary markdown report for the DisProt benchmark.

Usage:
    generate_report.py <all_disprot_metrics.parquet> <figures_dir> <output.md>
"""

import sys
from pathlib import Path
from datetime import date

import polars as pl


MYA = {
    "mouse": 100, "chicken": 300, "zebrafish": 430,
    "ciona": 550, "fly": 600, "worm": 650,
    "yeast": 900, "arabidopsis": 1500, "ecoli": 2000,
}

DISORDER_CATEGORIES = ["ordered", "partial", "disordered"]


def fmt(v) -> str:
    if v is None or v != v:   # nan check
        return "—"
    return f"{v:.3f}"


def main(metrics_parquet: str, figures_dir: str, output_md: str) -> None:
    metrics  = pl.read_parquet(metrics_parquet)
    fig_path = Path(figures_dir)
    today    = date.today().isoformat()

    # All-query summary
    all_metrics = metrics.filter(pl.col("disorder_category") == "all")
    n_queries = all_metrics.select(pl.col("n_positives").max()).item()

    # Per-tool, per-disorder-category recall@FDR5% (mean across species)
    strat = (
        metrics.filter(pl.col("disorder_category").is_in(DISORDER_CATEGORIES))
        .group_by(["tool", "disorder_category"])
        .agg(
            pl.col("recall_at_fdr05").mean().round(4).alias("mean_recall"),
            pl.col("auc_pr").mean().round(4).alias("mean_auc_pr"),
            pl.len().alias("n_species"),
        )
        .sort(["disorder_category", "tool"])
    )

    tools = sorted(strat["tool"].unique().to_list())

    # Build recall table: rows = disorder_category, cols = tools
    lines = []
    lines.append(f"# DisProt Benchmark Report")
    lines.append(f"\nGenerated: {today}\n")
    lines.append("## Overview\n")
    lines.append(f"- **Goal**: Measure whether Kmerseek detects homology through "
                 f"intrinsically disordered regions where Foldseek fails.")
    lines.append(f"- **Query set**: DisProt-annotated human proteins with ≥1 shared "
                 f"Pfam domain in ≥3 QfO species.")
    lines.append(f"- **Ground truth**: Reused from existing Pfam QfO benchmark "
                 f"(9 species, 100–2000 MYA).\n")

    lines.append("## Benchmark Size\n")
    lines.append(f"- DisProt query proteins: **{n_queries}** (approximate TP pairs across all species)")
    lines.append(f"- Species: 9 (mouse → E. coli, 100–2000 MYA)\n")

    lines.append("## Recall @ FDR 5% by Disorder Category\n")
    lines.append("Mean recall across all 9 species.\n")

    # Header
    header = "| Disorder category | " + " | ".join(tools) + " |"
    sep    = "|---|" + "---|" * len(tools)
    lines.append(header)
    lines.append(sep)

    for cat in DISORDER_CATEGORIES:
        row_vals = []
        for tool in tools:
            val = strat.filter(
                (pl.col("disorder_category") == cat) & (pl.col("tool") == tool)
            )["mean_recall"].to_list()
            row_vals.append(fmt(val[0]) if val else "—")
        cat_label = {
            "ordered": "Ordered (< 0.2)", "partial": "Partial (0.2–0.5)",
            "disordered": "Disordered (> 0.5)"
        }.get(cat, cat)
        lines.append(f"| {cat_label} | " + " | ".join(row_vals) + " |")

    lines.append("\n## AUC-PR (All Queries) — Mean Across Species\n")
    mean_auc = (
        all_metrics.group_by("tool")
        .agg(pl.col("auc_pr").mean().round(4).alias("mean_auc_pr"))
        .sort("mean_auc_pr", descending=True)
    )
    lines.append("| Tool | Mean AUC-PR |")
    lines.append("|---|---|")
    for row in mean_auc.iter_rows(named=True):
        lines.append(f"| {row['tool']} | {fmt(row['mean_auc_pr'])} |")

    lines.append("\n## Figures\n")
    figure_files = sorted(fig_path.glob("*.pdf"))
    if figure_files:
        for f in figure_files:
            lines.append(f"- [{f.name}]({fig_path}/{f.name})")
    else:
        lines.append("_Figures not yet generated._")

    lines.append("\n## Key Prediction\n")
    lines.append(
        "Foldseek recall should drop steeply as query disorder increases "
        "(no stable 3D structure → missing AlphaFold model or low confidence). "
        "Kmerseek recall should remain flat across disorder bins because k-mer "
        "matching is sequence-based and does not depend on 3D structure."
    )

    lines.append("\n## Interpretation Notes\n")
    lines.append("- `ordered`: mean metapredict score < 0.2 (structured proteins — "
                 "Foldseek expected to perform well here)")
    lines.append("- `partial`: 0.2–0.5 (mixed; contains both structured domains and "
                 "disordered loops)")
    lines.append("- `disordered`: > 0.5 (predominantly disordered — "
                 "Foldseek expected to fail; Kmerseek hypothesis: no drop)")
    lines.append("\nFoldseek gap = fraction of DisProt queries with no AlphaFold "
                 "structure and thus 0 recall for Foldseek regardless of threshold.")

    with open(output_md, "w") as fh:
        fh.write("\n".join(lines) + "\n")

    print(f"Wrote {output_md}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3])
