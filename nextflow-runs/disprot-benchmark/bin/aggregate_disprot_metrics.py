#!/usr/bin/env python3
"""
aggregate_disprot_metrics.py

Combine per-tool-per-species DisProt metrics into one table, summarise the kmerseek
alphabet x ksize sweep, and draw the figures.

Usage:
    aggregate_disprot_metrics.py <metrics_dir> <out_metrics.parquet> \\
        <out_pr_curves.parquet> <figures_dir> [--sweep-out sweep.parquet] [--headline-n 3]

Every tool is one row per species x disorder stratum in all_disprot_metrics.parquet. The
sweep summary averages each kmerseek (alphabet, ksize, mask) cell across the nine
species, per stratum.

Figures. "Headline tools" are the baselines plus the --headline-n kmerseek combos with
the highest mean recall at 5% FDR on the disordered stratum, which is the stratum this
benchmark exists for. The sweep figures show every combo.
  - sweep_recall_heatmap[_lc<mask>].pdf : alphabet x ksize, colour = mean recall@FDR5%,
                                          one panel per disorder stratum
  - sweep_best_ksize[_lc<mask>].pdf     : each alphabet at its best ksize, against the
                                          MMseqs2 baseline
  - recall_vs_disorder.pdf              : headline tools, recall@FDR5% per stratum
  - recall_by_divergence.pdf            : headline tools, recall vs Mya per stratum
  - auc_heatmap.pdf                     : headline tools x species, AUC-PR
  - pr_curves_by_disorder.pdf           : headline tools, PR curves per stratum
"""

import argparse
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from disprot_tool_names import alphabet_classes, annotate, headline_tools, is_kmerseek  # noqa: E402

MYA = {
    "mouse": 100, "chicken": 300, "zebrafish": 430,
    "ciona": 550, "fly": 600, "worm": 650,
    "yeast": 900, "arabidopsis": 1500, "ecoli": 2000,
}

DISORDER_CATEGORY_ORDER = ["ordered", "partial", "disordered", "all"]
STRATA = ["ordered", "partial", "disordered"]
DISORDER_LABELS = {
    "ordered":    "Ordered queries\n(mean disorder < 0.2)",
    "partial":    "Partly disordered\n(0.2 to 0.5)",
    "disordered": "Disordered queries\n(> 0.5)",
    "all":        "All queries",
}

# One hue per tool family; the combo is told apart by its label, marker and line style,
# never by a second shade of the same hue.
FAMILY_COLORS = {"mmseqs2": "#2ca02c", "foldseek": "#d62728", "kmerseek": "#1f77b4"}
KMERSEEK_MARKERS = ["o", "s", "^", "D", "v", "P"]
KMERSEEK_STYLES = ["-", "--", ":", "-.", (0, (5, 1)), (0, (1, 1))]


def tool_family(tool: str) -> str:
    return "kmerseek" if is_kmerseek(tool) else tool


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("metrics_dir")
    p.add_argument("out_metrics")
    p.add_argument("out_pr_curves")
    p.add_argument("figures_dir")
    p.add_argument("--sweep-out", default=None)
    p.add_argument("--headline-n", type=int, default=3)
    args = p.parse_args()

    metrics_path = Path(args.metrics_dir)
    fig_path = Path(args.figures_dir)
    fig_path.mkdir(parents=True, exist_ok=True)

    metric_files = sorted(metrics_path.glob("*.disprot_metrics.parquet"))
    pr_curve_files = sorted(metrics_path.glob("*.disprot_pr_curve.parquet"))
    if not metric_files:
        sys.exit(f"No *.disprot_metrics.parquet files found in {metrics_path}")

    # ------------------------------------------------------------------
    # Combine and annotate
    # ------------------------------------------------------------------
    metrics = pl.concat([pl.read_parquet(f) for f in metric_files], how="diagonal_relaxed")
    # The evaluator writes NaN for a stratum with no positives. NaN is a value to polars,
    # not a gap: a mean over it is NaN and drop_nulls keeps it. Make it a gap.
    metrics = metrics.with_columns(
        [pl.col(c).fill_nan(None) for c in metrics.columns if metrics[c].dtype in (pl.Float32, pl.Float64)]
    )
    metrics = annotate(metrics).with_columns(
        pl.col("species").replace_strict(MYA, default=None).cast(pl.Int32).alias("mya")
    )
    cat_ord = {c: i for i, c in enumerate(DISORDER_CATEGORY_ORDER)}
    metrics = (
        metrics.with_columns(
            pl.col("disorder_category").replace_strict(cat_ord, default=99).alias("_cat_order")
        )
        .sort(["_cat_order", "tool", "mya"])
        .drop("_cat_order")
    )
    metrics.write_parquet(args.out_metrics)

    if pr_curve_files:
        pr_curves = pl.concat([pl.read_parquet(f) for f in pr_curve_files], how="diagonal_relaxed")
        pr_curves = pr_curves.with_columns(
            pl.col("species").replace_strict(MYA, default=None).cast(pl.Int32).alias("mya")
        )
    else:
        pr_curves = pl.DataFrame(
            schema={"tool": pl.String, "species": pl.String, "disorder_category": pl.String,
                    "precision": pl.Float64, "recall": pl.Float64, "threshold": pl.Float64,
                    "mya": pl.Int32}
        )
    pr_curves.write_parquet(args.out_pr_curves)

    # ------------------------------------------------------------------
    # Sweep summary: mean over species per (alphabet, ksize, mask, stratum)
    # ------------------------------------------------------------------
    sweep = (
        metrics.filter(pl.col("alphabet").is_not_null())
        .group_by(["tool", "alphabet", "ksize", "lowcomp", "disorder_category"])
        .agg(
            pl.col("recall_at_fdr05").mean().alias("mean_recall_fdr05"),
            pl.col("auc_pr").mean().alias("mean_auc_pr"),
            pl.col("n_found").sum().alias("n_found"),
            pl.col("n_positives").sum().alias("n_positives"),
            pl.col("species").n_unique().alias("n_species"),
        )
        .with_columns(
            pl.col("alphabet").map_elements(alphabet_classes, return_dtype=pl.Int32).alias("classes")
        )
        .sort(["disorder_category", "classes", "alphabet", "ksize", "lowcomp"], descending=[False, True, False, False, False])
    )
    if args.sweep_out:
        sweep.write_parquet(args.sweep_out)

    headline = headline_tools(metrics, args.headline_n)

    # ------------------------------------------------------------------
    # Console tables: the numbers every figure below is drawn from
    # ------------------------------------------------------------------
    pl.Config.set_tbl_rows(60)
    pl.Config.set_tbl_cols(12)
    print("\n=== kmerseek sweep: best ksize per alphabet, mean recall@FDR5% across species ===\n")
    best = (
        sweep.filter(pl.col("disorder_category") == "disordered")
        .sort("mean_recall_fdr05", descending=True)
        .group_by(["alphabet", "lowcomp"], maintain_order=True)
        .first()
        .select(["alphabet", "classes", "lowcomp", "ksize", "mean_recall_fdr05", "mean_auc_pr", "n_species"])
        .sort(["lowcomp", "mean_recall_fdr05"], descending=[False, True])
    )
    print(best)

    print(f"\n=== headline tools (baselines + top {args.headline_n} kmerseek combos on the disordered stratum) ===\n")
    print(headline)

    print("\n=== recall@FDR5% by headline tool and disorder category (mean across species) ===\n")
    print(
        metrics.filter(pl.col("tool").is_in(headline) & (pl.col("disorder_category") != "all"))
        .group_by(["tool", "disorder_category"])
        .agg(pl.col("recall_at_fdr05").mean().round(4).alias("mean_recall_fdr05"))
        .sort(["disorder_category", "mean_recall_fdr05"], descending=[False, True])
    )

    print("\n=== AUC-PR (all queries) by headline tool and species ===\n")
    all_only = metrics.filter((pl.col("disorder_category") == "all") & pl.col("tool").is_in(headline))
    pivot = all_only.select(["tool", "species", "auc_pr"]).pivot(
        on="species", index="tool", values="auc_pr", aggregate_function="first"
    )
    species_by_mya = sorted(MYA, key=MYA.get)
    print(pivot.select(["tool"] + [s for s in species_by_mya if s in pivot.columns]))

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------
    try:
        make_sweep_figures(sweep, metrics, fig_path)
        make_headline_figures(metrics, pr_curves, headline, fig_path)
    except ImportError as e:
        print(f"WARNING: matplotlib not available, skipping figures: {e}")

    print(f"\nWrote {args.out_metrics}, {args.out_pr_curves}, "
          f"{args.sweep_out or '(no sweep summary)'}, figures in {fig_path}/")


def legend_above(fig, axes, title: str) -> None:
    """Title on top, the legend under it, the panels under that: legend before marks in
    reading order, and nothing overlapping whatever the panel count."""
    handles, labels = axes[0].get_legend_handles_labels()
    fig.tight_layout(rect=[0, 0, 1, 0.84])
    fig.legend(handles, labels, loc="upper center", ncol=min(5, max(1, len(labels))), fontsize=7,
               frameon=False, bbox_to_anchor=(0.5, 0.92))
    fig.suptitle(title, y=0.99, fontsize=10)


def make_sweep_figures(sweep: pl.DataFrame, metrics: pl.DataFrame, fig_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import numpy as np

    if len(sweep) == 0:
        print("no kmerseek combos in this run; skipping sweep figures")
        return

    masks = sorted(sweep["lowcomp"].unique().to_list())
    # A mask suffix only when both settings ran, so the single-setting run keeps the
    # plain filename.
    suffix = {lc: (f"_lc{str(lc).lower()}" if len(masks) > 1 else "") for lc in masks}

    baseline = (
        metrics.filter(pl.col("tool") == "mmseqs2")
        .group_by("disorder_category")
        .agg(pl.col("recall_at_fdr05").mean().alias("r"))
    )
    baseline = dict(zip(baseline["disorder_category"], baseline["r"]))

    for lc in masks:
        sub = sweep.filter(pl.col("lowcomp") == lc)
        alphabets = (
            sub.select(["alphabet", "classes"]).unique()
            .sort(["classes", "alphabet"], descending=[True, False])["alphabet"].to_list()
        )
        ksizes = list(range(int(sub["ksize"].min()), int(sub["ksize"].max()) + 1))
        mask_note = "low-complexity k-mers removed" if lc else "low-complexity k-mers kept"

        # -- Heatmap: alphabet x ksize per stratum. Colour has one meaning: recall. --
        fig, axes = plt.subplots(1, len(STRATA), figsize=(5.2 * len(STRATA), 0.38 * len(alphabets) + 2.2),
                                 sharey=True)
        for ax, cat in zip(axes, STRATA):
            grid = np.full((len(alphabets), len(ksizes)), np.nan)
            cell = sub.filter(pl.col("disorder_category") == cat)
            for a, k, r in cell.select(["alphabet", "ksize", "mean_recall_fdr05"]).iter_rows():
                if r is not None:
                    grid[alphabets.index(a), ksizes.index(k)] = r
            im = ax.imshow(grid, aspect="auto", cmap="Blues", vmin=0, vmax=1)
            ax.set_xticks(range(len(ksizes)))
            ax.set_xticklabels(ksizes, fontsize=7)
            ax.set_xlabel("k-mer size k (residues)")
            ax.set_title(DISORDER_LABELS[cat], fontsize=10)
            for i in range(len(alphabets)):
                for j in range(len(ksizes)):
                    if not np.isnan(grid[i, j]):
                        ax.text(j, i, f"{grid[i, j]*100:.0f}", ha="center", va="center",
                                fontsize=5.5, color="white" if grid[i, j] > 0.6 else "black")
        axes[0].set_yticks(range(len(alphabets)))
        axes[0].set_yticklabels([f"{a} ({alphabet_classes(a)} classes)" for a in alphabets], fontsize=7)
        axes[0].set_ylabel("Alphabet, finest to coarsest")
        cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01)
        cbar.set_label("Recall at 5% FDR, mean over 9 species")
        cbar.ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        fig.suptitle(f"kmerseek alphabet x ksize sweep on DisProt queries ({mask_note}); "
                     "cell numbers are recall in %, blank = not run", fontsize=11, y=1.02)
        fig.savefig(fig_path / f"sweep_recall_heatmap{suffix[lc]}.pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"Saved sweep_recall_heatmap{suffix[lc]}.pdf")

        # -- Best ksize per alphabet, against MMseqs2 --
        fig, axes = plt.subplots(1, len(STRATA), figsize=(4.4 * len(STRATA), 0.32 * len(alphabets) + 2.0),
                                 sharey=True)
        for ax, cat in zip(axes, STRATA):
            cell = (
                sub.filter(pl.col("disorder_category") == cat)
                .sort("mean_recall_fdr05", descending=True, nulls_last=True)
                .group_by("alphabet", maintain_order=True).first()
            )
            rows = {a: (k, r) for a, k, r in cell.select(["alphabet", "ksize", "mean_recall_fdr05"]).iter_rows()}
            ys = range(len(alphabets))
            xs = [rows.get(a, (None, np.nan))[1] for a in alphabets]
            ax.scatter(xs, ys, color=FAMILY_COLORS["kmerseek"], marker="o", zorder=3,
                       label="kmerseek, alphabet at its best k")
            for y, a in zip(ys, alphabets):
                k, r = rows.get(a, (None, None))
                if r is not None and not np.isnan(r):
                    ax.text(r + 0.02, y, f"k={k}", va="center", fontsize=6)
            if cat in baseline and baseline[cat] is not None:
                ax.axvline(baseline[cat], color="#555555", linestyle="--", linewidth=1,
                           label="MMseqs2 (-s 7), same queries")
            ax.set_xlim(0, 1.08)
            ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
            ax.set_xlabel("Recall at 5% FDR, mean over 9 species")
            ax.set_title(DISORDER_LABELS[cat], fontsize=10)
            ax.grid(axis="x", alpha=0.3)
        axes[0].set_yticks(list(range(len(alphabets))))
        axes[0].set_yticklabels([f"{a} ({alphabet_classes(a)} classes)" for a in alphabets], fontsize=7)
        axes[0].invert_yaxis()
        legend_above(fig, axes, f"Each alphabet at its best k on DisProt queries ({mask_note})")
        fig.savefig(fig_path / f"sweep_best_ksize{suffix[lc]}.pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"Saved sweep_best_ksize{suffix[lc]}.pdf")


def make_headline_figures(metrics: pl.DataFrame, pr_curves: pl.DataFrame,
                          headline: list[str], fig_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import numpy as np

    km_tools = [t for t in headline if is_kmerseek(t)]
    style = {}
    for t in headline:
        fam = tool_family(t)
        i = km_tools.index(t) if fam == "kmerseek" else 0
        style[t] = dict(color=FAMILY_COLORS.get(fam, "#888888"),
                        marker=KMERSEEK_MARKERS[i % len(KMERSEEK_MARKERS)] if fam == "kmerseek" else "o",
                        linestyle=KMERSEEK_STYLES[i % len(KMERSEEK_STYLES)] if fam == "kmerseek" else "-")

    # -- Figure 1: recall@FDR5% by stratum, bars per headline tool --
    fig, axes = plt.subplots(1, len(STRATA), figsize=(4.2 * len(STRATA), 4.2), sharey=True)
    for ax, cat in zip(axes, STRATA):
        sub = metrics.filter(pl.col("disorder_category") == cat)
        for i, tool in enumerate(headline):
            vals = sub.filter(pl.col("tool") == tool)["recall_at_fdr05"].drop_nulls().to_numpy()
            if len(vals) == 0:
                continue
            mean = float(np.nanmean(vals))
            sem = float(np.nanstd(vals) / max(1, np.sqrt(len(vals))))
            ax.bar(i, mean, yerr=sem, color=style[tool]["color"], capsize=4, alpha=0.85)
            ax.text(i, mean + sem + 0.02, f"{mean*100:.0f}%", ha="center", fontsize=7)
        ax.set_title(DISORDER_LABELS[cat], fontsize=10)
        ax.set_xticks(range(len(headline)))
        ax.set_xticklabels(headline, rotation=35, ha="right", fontsize=7)
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    axes[0].set_ylabel("Recall at 5% FDR, mean over 9 species (bar = s.e.m.)")
    fig.suptitle("DisProt benchmark: recall by query disorder. Colour = tool family "
                 "(green MMseqs2, red Foldseek, blue kmerseek)", fontsize=10)
    fig.tight_layout()
    fig.savefig(fig_path / "recall_vs_disorder.pdf")
    plt.close(fig)
    print("Saved recall_vs_disorder.pdf")

    # -- Figure 2: recall vs Mya, faceted by stratum --
    fig, axes = plt.subplots(1, len(STRATA), figsize=(4.6 * len(STRATA), 4.2), sharey=True)
    for ax, cat in zip(axes, STRATA):
        sub = metrics.filter(pl.col("disorder_category") == cat).sort("mya")
        for tool in headline:
            td = sub.filter(pl.col("tool") == tool)
            if len(td) == 0:
                continue
            ax.plot(td["mya"].to_list(), td["recall_at_fdr05"].to_list(), label=tool,
                    linewidth=1.5, markersize=4, **style[tool])
        ax.set_title(DISORDER_LABELS[cat], fontsize=10)
        ax.set_xlabel("Divergence from human (Mya, log scale)")
        ax.set_ylim(0, 1)
        ax.set_xscale("log")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    axes[0].set_ylabel("Recall at 5% FDR")
    legend_above(fig, axes, "DisProt benchmark: recall against divergence, one panel per query disorder level")
    fig.savefig(fig_path / "recall_by_divergence.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved recall_by_divergence.pdf")

    # -- Figure 3: AUC-PR heatmap, headline tools x species --
    all_only = metrics.filter(pl.col("disorder_category") == "all")
    tools = [t for t in headline if t in all_only["tool"].unique().to_list()]
    species = sorted(MYA, key=MYA.get)
    data = np.full((len(tools), len(species)), np.nan)
    for i, tool in enumerate(tools):
        for j, sp in enumerate(species):
            vals = all_only.filter((pl.col("tool") == tool) & (pl.col("species") == sp))["auc_pr"].to_list()
            if vals and vals[0] is not None:
                data[i, j] = vals[0]
    fig, ax = plt.subplots(figsize=(10, len(tools) * 0.6 + 1.5))
    im = ax.imshow(data, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(species)))
    ax.set_xticklabels([f"{s}\n{MYA[s]} Mya" for s in species], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(tools)))
    ax.set_yticklabels(tools, fontsize=8)
    for i in range(len(tools)):
        for j in range(len(species)):
            if not np.isnan(data[i, j]):
                ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if data[i, j] > 0.6 else "black")
    plt.colorbar(im, ax=ax, label="AUC-PR, all DisProt queries")
    ax.set_title("DisProt benchmark: area under the precision-recall curve, by tool and target species")
    fig.tight_layout()
    fig.savefig(fig_path / "auc_heatmap.pdf")
    plt.close(fig)
    print("Saved auc_heatmap.pdf")

    # -- Figure 4: PR curves by stratum --
    if len(pr_curves) > 0:
        fig, axes = plt.subplots(1, len(STRATA), figsize=(4.6 * len(STRATA), 4.2), sharey=True)
        for ax, cat in zip(axes, STRATA):
            sub = pr_curves.filter(pl.col("disorder_category") == cat)
            for tool in headline:
                td = sub.filter(pl.col("tool") == tool)
                if len(td) == 0:
                    continue
                ax.plot(td["recall"].to_list(), td["precision"].to_list(), label=tool,
                        linewidth=1.2, alpha=0.85, color=style[tool]["color"],
                        linestyle=style[tool]["linestyle"])
            ax.set_title(DISORDER_LABELS[cat], fontsize=10)
            ax.set_xlabel("Recall")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
        axes[0].set_ylabel("Precision")
        legend_above(fig, axes, "DisProt benchmark: precision-recall curves, every species pooled")
        fig.savefig(fig_path / "pr_curves_by_disorder.pdf", bbox_inches="tight")
        plt.close(fig)
        print("Saved pr_curves_by_disorder.pdf")


if __name__ == "__main__":
    main()
