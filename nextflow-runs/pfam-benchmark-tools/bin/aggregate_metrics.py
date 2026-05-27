#!/Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3
"""
aggregate_metrics.py

Combine per-tool-per-species metrics into summary tables and print a comparison.

Usage (standalone):
    python aggregate_metrics.py <results_dir> <out_metrics.parquet> <out_pr_curves.parquet>

Called automatically by main.nf's aggregateMetrics process.
"""

import sys
from pathlib import Path

import polars as pl

# Evolutionary distance for each species (MYA from human)
MYA = {
    "mouse": 100,
    "chicken": 300,
    "zebrafish": 430,
    "ciona": 550,
    "fly": 600,
    "worm": 650,
    "yeast": 900,
    "arabidopsis": 1500,
    "ecoli": 2000,
}

# Display order for tools (sensitivity ladder)
TOOL_ORDER = [
    "diamond",
    "mmseqs2_seqseq",
    "hmmer3_phmmer",
    "mmseqs2_iterative",
    "hmmer3_hmmscan",
    "hhblits",
    "psalm",
]


def main(results_dir: str, out_metrics: str, out_pr_curves: str) -> None:
    results_path = Path(results_dir)

    metric_files   = sorted(results_path.glob("*.metrics.parquet"))
    pr_curve_files = sorted(results_path.glob("*.pr_curve.parquet"))

    if not metric_files:
        print(f"No *.metrics.parquet files found in {results_path}")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Combine metrics
    # -----------------------------------------------------------------------
    metrics = pl.concat([pl.read_parquet(f) for f in metric_files])
    metrics = metrics.with_columns(
        pl.col("species").replace(MYA).cast(pl.Int32).alias("mya")
    )

    tool_order = {t: i for i, t in enumerate(TOOL_ORDER)}
    metrics = metrics.with_columns(
        pl.col("tool")
        .map_elements(lambda t: tool_order.get(t, 99), return_dtype=pl.Int32)
        .alias("_tool_order")
    )
    metrics = metrics.sort(["_tool_order", "mya"]).drop("_tool_order")
    metrics.write_parquet(out_metrics)

    # -----------------------------------------------------------------------
    # Combine PR curves
    # -----------------------------------------------------------------------
    if pr_curve_files:
        pr_curves = pl.concat([pl.read_parquet(f) for f in pr_curve_files])
        pr_curves = pr_curves.with_columns(
            pl.col("species").replace(MYA).cast(pl.Int32).alias("mya")
        )
        pr_curves.write_parquet(out_pr_curves)
    else:
        pl.DataFrame(
            {"tool": [], "species": [], "precision": [], "recall": [], "threshold": [], "mya": []}
        ).write_parquet(out_pr_curves)

    # -----------------------------------------------------------------------
    # Print AUC-PR table (tools × species)
    # -----------------------------------------------------------------------
    print("\n=== AUC-PR by tool and species (MYA) ===\n")
    pivot = (
        metrics
        .select(["tool", "species", "mya", "auc_pr"])
        .sort(["tool", "mya"])
        .pivot(on="species", index="tool", values="auc_pr", aggregate_function="first")
    )
    # Sort species columns by MYA
    species_by_mya = sorted(MYA.keys(), key=lambda s: MYA[s])
    cols = ["tool"] + [s for s in species_by_mya if s in pivot.columns]
    print(pivot.select(cols))

    print("\n=== AUC-PR: mean across all species ===\n")
    mean_auc = (
        metrics
        .group_by("tool")
        .agg(pl.col("auc_pr").mean().round(4).alias("mean_auc_pr"))
        .sort("mean_auc_pr", descending=True)
    )
    print(mean_auc)

    print(f"\nWrote {out_metrics} and {out_pr_curves}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3])
