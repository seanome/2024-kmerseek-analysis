#!/Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3
"""
aggregate_disprot_metrics.py

Combine per-tool-per-species DisProt metrics and produce summary figures.

Usage:
    aggregate_disprot_metrics.py <metrics_dir> <out_metrics.parquet> \\
        <out_pr_curves.parquet> <figures_dir>

Figures produced:
  - recall_vs_disorder.pdf        : recall@FDR5% vs mean disorder (key plot)
  - recall_by_divergence.pdf      : recall vs MYA, faceted by disorder_category
  - foldseek_gap.pdf              : fraction of DisProt queries unsearchable by Foldseek
  - pr_curves_by_disorder.pdf     : PR curves per disorder category
  - auc_heatmap.pdf               : AUC-PR heatmap (tools × species)
"""

import sys
from pathlib import Path

import polars as pl


MYA = {
    "mouse": 100, "chicken": 300, "zebrafish": 430,
    "ciona": 550, "fly": 600, "worm": 650,
    "yeast": 900, "arabidopsis": 1500, "ecoli": 2000,
}

DISORDER_CATEGORY_ORDER = ["ordered", "partial", "disordered", "all"]

TOOL_ORDER = [
    "mmseqs2",
    "foldseek",
    "kmerseek_k20",
    "kmerseek_k24",
    "kmerseek_k27",
]

TOOL_COLORS = {
    "mmseqs2":      "#2ca02c",
    "foldseek":     "#d62728",
    "kmerseek_k20": "#9ecae1",
    "kmerseek_k24": "#1f77b4",
    "kmerseek_k27": "#08306b",
}


def main(metrics_dir: str, out_metrics: str, out_pr_curves: str, figures_dir: str) -> None:
    metrics_path = Path(metrics_dir)
    fig_path     = Path(figures_dir)
    fig_path.mkdir(parents=True, exist_ok=True)

    metric_files   = sorted(metrics_path.glob("*.disprot_metrics.parquet"))
    pr_curve_files = sorted(metrics_path.glob("*.disprot_pr_curve.parquet"))

    if not metric_files:
        print(f"No *.disprot_metrics.parquet files found in {metrics_path}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Combine and annotate
    # ------------------------------------------------------------------
    metrics = pl.concat([pl.read_parquet(f) for f in metric_files])
    metrics = metrics.with_columns(
        pl.col("species").replace(MYA).cast(pl.Int32).alias("mya")
    )

    tool_ord = {t: i for i, t in enumerate(TOOL_ORDER)}
    metrics = metrics.with_columns(
        pl.col("tool").map_elements(lambda t: tool_ord.get(t, 99), return_dtype=pl.Int32)
            .alias("_tool_order"),
        pl.col("disorder_category").map_elements(
            lambda c: DISORDER_CATEGORY_ORDER.index(c) if c in DISORDER_CATEGORY_ORDER else 99,
            return_dtype=pl.Int32
        ).alias("_cat_order"),
    ).sort(["_cat_order", "_tool_order", "mya"]).drop(["_tool_order", "_cat_order"])

    metrics.write_parquet(out_metrics)

    if pr_curve_files:
        pr_curves = pl.concat([pl.read_parquet(f) for f in pr_curve_files])
        pr_curves = pr_curves.with_columns(
            pl.col("species").replace(MYA).cast(pl.Int32).alias("mya")
        )
        pr_curves.write_parquet(out_pr_curves)
    else:
        pl.DataFrame({
            "tool": [], "species": [], "disorder_category": [],
            "precision": [], "recall": [], "threshold": [], "mya": [],
        }).write_parquet(out_pr_curves)

    # ------------------------------------------------------------------
    # Console summary tables
    # ------------------------------------------------------------------
    print("\n=== recall@FDR5% by tool and disorder category (mean across species) ===\n")
    summary = (
        metrics.filter(pl.col("disorder_category") != "all")
        .group_by(["tool", "disorder_category"])
        .agg(pl.col("recall_at_fdr05").mean().round(4).alias("mean_recall_fdr05"))
        .sort(["disorder_category", "mean_recall_fdr05"], descending=[False, True])
    )
    print(summary)

    print("\n=== AUC-PR (all queries) by tool and species ===\n")
    all_only = metrics.filter(pl.col("disorder_category") == "all")
    pivot = (
        all_only.select(["tool", "species", "auc_pr"])
        .pivot(on="species", index="tool", values="auc_pr", aggregate_function="first")
    )
    species_by_mya = sorted(MYA.keys(), key=lambda s: MYA[s])
    cols = ["tool"] + [s for s in species_by_mya if s in pivot.columns]
    print(pivot.select(cols))

    # ------------------------------------------------------------------
    # Figures (matplotlib)
    # ------------------------------------------------------------------
    try:
        _make_figures(metrics, pr_curves if pr_curve_files else None, fig_path)
    except ImportError as e:
        print(f"WARNING: matplotlib not available, skipping figures: {e}")

    print(f"\nWrote {out_metrics}, {out_pr_curves}, figures in {figures_dir}/")


def _make_figures(metrics, pr_curves, fig_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import numpy as np

    DISORDER_LABELS = {
        "ordered":   "Ordered\n(disorder < 0.2)",
        "partial":   "Partial\n(0.2 – 0.5)",
        "disordered": "Disordered\n(> 0.5)",
        "all":       "All queries",
    }

    # ── Figure 1: recall@FDR5% by disorder category (bar chart per tool) ──
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for ax, cat in zip(axes, ["ordered", "partial", "disordered"]):
        sub = metrics.filter(pl.col("disorder_category") == cat)
        for i, tool in enumerate(TOOL_ORDER):
            tool_data = sub.filter(pl.col("tool") == tool)
            if len(tool_data) == 0:
                continue
            vals = tool_data["recall_at_fdr05"].to_list()
            mean = np.nanmean(vals)
            sem  = np.nanstd(vals) / max(1, np.sqrt(len(vals)))
            ax.bar(i, mean, yerr=sem, color=TOOL_COLORS.get(tool, "#888"),
                   capsize=4, label=tool, alpha=0.85)
        ax.set_title(DISORDER_LABELS[cat])
        ax.set_xticks(range(len(TOOL_ORDER)))
        ax.set_xticklabels(TOOL_ORDER, rotation=30, ha="right", fontsize=7)
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    axes[0].set_ylabel("Recall @ FDR 5%")
    axes[1].set_xlabel("Tool")
    fig.suptitle("DisProt Benchmark: Recall by Disorder Category", fontsize=12)
    fig.tight_layout()
    fig.savefig(fig_path / "recall_vs_disorder.pdf")
    plt.close(fig)
    print("Saved recall_vs_disorder.pdf")

    # ── Figure 2: recall vs MYA, faceted by disorder category ──
    cats_to_plot = ["ordered", "partial", "disordered"]
    fig, axes = plt.subplots(1, len(cats_to_plot), figsize=(14, 4), sharey=True)
    for ax, cat in zip(axes, cats_to_plot):
        sub = metrics.filter(pl.col("disorder_category") == cat).sort("mya")
        for tool in TOOL_ORDER:
            tool_data = sub.filter(pl.col("tool") == tool)
            if len(tool_data) == 0:
                continue
            mya_vals = tool_data["mya"].to_list()
            rec_vals = tool_data["recall_at_fdr05"].to_list()
            ax.plot(mya_vals, rec_vals, "o-",
                    color=TOOL_COLORS.get(tool, "#888"), label=tool, linewidth=1.5)
        ax.set_title(DISORDER_LABELS[cat])
        ax.set_xlabel("Evolutionary distance (MYA)")
        ax.set_ylim(0, 1)
        ax.set_xscale("log")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    axes[0].set_ylabel("Recall @ FDR 5%")
    axes[0].legend(fontsize=7)
    fig.suptitle("DisProt Benchmark: Recall vs Evolutionary Distance", fontsize=12)
    fig.tight_layout()
    fig.savefig(fig_path / "recall_by_divergence.pdf")
    plt.close(fig)
    print("Saved recall_by_divergence.pdf")

    # ── Figure 3: AUC-PR heatmap ──
    all_only = metrics.filter(pl.col("disorder_category") == "all")
    tools    = [t for t in TOOL_ORDER if t in all_only["tool"].unique().to_list()]
    species  = sorted(MYA.keys(), key=lambda s: MYA[s])

    data = np.full((len(tools), len(species)), np.nan)
    for i, tool in enumerate(tools):
        for j, sp in enumerate(species):
            vals = all_only.filter(
                (pl.col("tool") == tool) & (pl.col("species") == sp)
            )["auc_pr"].to_list()
            if vals:
                data[i, j] = vals[0]

    fig, ax = plt.subplots(figsize=(10, len(tools) * 0.7 + 1))
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(species)))
    ax.set_xticklabels([f"{s}\n{MYA[s]}MYA" for s in species], rotation=45, ha="right")
    ax.set_yticks(range(len(tools)))
    ax.set_yticklabels(tools)
    for i in range(len(tools)):
        for j in range(len(species)):
            if not np.isnan(data[i, j]):
                ax.text(j, i, f"{data[i,j]:.2f}", ha="center", va="center", fontsize=7)
    plt.colorbar(im, ax=ax, label="AUC-PR")
    ax.set_title("DisProt Benchmark: AUC-PR by Tool and Species")
    fig.tight_layout()
    fig.savefig(fig_path / "auc_heatmap.pdf")
    plt.close(fig)
    print("Saved auc_heatmap.pdf")

    # ── Figure 4: PR curves by disorder category (all-species mean) ──
    if pr_curves is not None and len(pr_curves) > 0:
        fig, axes = plt.subplots(1, len(cats_to_plot), figsize=(14, 4), sharey=True)
        for ax, cat in zip(axes, cats_to_plot):
            sub = pr_curves.filter(pl.col("disorder_category") == cat)
            for tool in TOOL_ORDER:
                tool_data = sub.filter(pl.col("tool") == tool)
                if len(tool_data) == 0:
                    continue
                rec  = tool_data["recall"].to_list()
                prec = tool_data["precision"].to_list()
                ax.plot(rec, prec, color=TOOL_COLORS.get(tool, "#888"),
                        label=tool, linewidth=1.5, alpha=0.85)
            ax.set_title(DISORDER_LABELS[cat])
            ax.set_xlabel("Recall")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
        axes[0].set_ylabel("Precision")
        axes[0].legend(fontsize=7)
        fig.suptitle("DisProt Benchmark: Precision-Recall Curves", fontsize=12)
        fig.tight_layout()
        fig.savefig(fig_path / "pr_curves_by_disorder.pdf")
        plt.close(fig)
        print("Saved pr_curves_by_disorder.pdf")


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
