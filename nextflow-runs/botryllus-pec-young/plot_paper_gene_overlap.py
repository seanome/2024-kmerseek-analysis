#!/usr/bin/env python3
"""Draw how many of the paper's listed genes this run found, and whether the two agree.

A listed gene in the 13-49 d comparison counts as found if either matching window here
(13-17 d, 49-52 d) finds it. Each listed entry falls in exactly one group below.

Usage: plot_paper_gene_overlap.py [<dir> <run>]   (run in this folder, after compare_paper_genes.py)
<dir> holds compare_paper_genes.py's output (default: this folder) and <run> is the run name
given to it (default: joint). Writes <dir>/paper_gene_summary.tsv and
figures/paper_gene_overlap[_<run>]_domen2026_dataset_s1.{pdf,svg,png}.
"""

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "notebooks"))
import pubfig as pf  # noqa: E402
from compare_paper_genes import RUN_WINDOWS  # noqa: E402

COMPARISONS = ["2 h", "24 h", "13-49 d", "132 d"]
WINDOW_LABELS = {
    "young_2h": "2 h",
    "young_24h": "24 h",
    "young_13_17d": "13-17 d",
    "young_21_28d": "21-28 d",
    "young_49_52d": "49-52 d",
    "young_104_132d": "104-132 d",
    "young_13_49d": "13-49 d",
    "young_132d": "132 d",
}
GROUPS = {
    "no_match_or_not_tested": ("No matching gene here, or not tested", pf.GREY),
    "tested_not_found": ("Tested, p >= 0.05 or opposite direction", "#DDDDDD"),
    "p_lt_0_05_same_direction": (
        "Unadjusted p < 0.05, same direction",
        pf.OKABE_ITO["blue"],
    ),
    "padj_lt_0_05_same_direction": (
        "Adjusted p < 0.05, same direction",
        pf.OKABE_ITO["vermillion"],
    ),
}


def summary(folder):
    genes = pl.read_csv(
        folder / "paper_gene_overlap_genes.tsv",
        separator="\t",
        null_values=["NA", ""],
        infer_schema_length=None,
    ).with_columns(
        same=pl.when(pl.col("direction") == "up")
        .then(pl.col("log2FoldChange") > 0)
        .otherwise(pl.col("log2FoldChange") < 0)
        .fill_null(False),
        tested=pl.col("matched") & pl.col("pvalue").is_not_null(),
    )
    genes = genes.with_columns(
        p_same=pl.col("tested") & pl.col("same") & (pl.col("pvalue") < 0.05),
        padj_same=(
            pl.col("tested") & pl.col("same") & (pl.col("padj") < 0.05)
        ).fill_null(False),
    )
    per_entry = genes.group_by("paper_gene", "comparison", "direction").agg(
        tested=pl.col("tested").any(),
        p_same=pl.col("p_same").any(),
        padj_same=pl.col("padj_same").any(),
    )
    table = (
        per_entry.group_by("comparison")
        .agg(
            listed=pl.len(),
            no_match_or_not_tested=(~pl.col("tested")).sum(),
            tested_not_found=(pl.col("tested") & ~pl.col("p_same")).sum(),
            p_lt_0_05_same_direction=(pl.col("p_same") & ~pl.col("padj_same")).sum(),
            padj_lt_0_05_same_direction=pl.col("padj_same").sum(),
        )
        .sort(
            pl.col("comparison").replace_strict(
                {c: i for i, c in enumerate(COMPARISONS)}
            )
        )
    )
    total = table.select(
        pl.lit("all").alias("comparison"), pl.exclude("comparison").sum()
    )
    return pl.concat([table, total])


def main(folder=".", run="joint"):
    folder = Path(folder)
    time_matched = {(c, w) for c, ws in RUN_WINDOWS[run].items() for w in ws}
    table = summary(folder)
    table.write_csv(folder / "paper_gene_summary.tsv", separator="\t")
    print(table)
    crossed = pl.read_csv(
        folder / "paper_gene_agreement_all_windows.tsv", separator="\t"
    )

    pf.use_style()
    fig, (left, right) = pf.figure(pf.TWO_COLUMN_MM, 72, ncols=2, width_ratios=[1, 1])

    rows = table.filter(pl.col("comparison") != "all").to_dicts()
    y = np.arange(len(rows))
    start = np.zeros(len(rows))
    for key, (label, colour) in GROUPS.items():
        share = np.array([100 * r[key] / r["listed"] for r in rows])
        left.barh(y, share, left=start, color=colour, label=label)
        start += share
    for i, r in enumerate(rows):
        found = r["p_lt_0_05_same_direction"] + r["padj_lt_0_05_same_direction"]
        left.text(101, i, f"{found} of {r['listed']}", va="center")
    left.set_yticks(y, [r["comparison"] for r in rows])
    left.invert_yaxis()
    left.set_xlim(0, 100)
    left.set_xlabel("Paper's listed genes (%)")
    left.set_ylabel("Paper comparison")
    left.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False)

    windows = [w for w in WINDOW_LABELS if w in set(crossed["window"])]
    grid = np.array(
        [
            [
                crossed.filter((pl.col("comparison") == c) & (pl.col("window") == w))[
                    "spearman"
                ][0]
                for w in windows
            ]
            for c in COMPARISONS
        ]
    )
    limit = np.abs(grid).max()
    image = right.imshow(grid, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    for i, c in enumerate(COMPARISONS):
        for j, w in enumerate(windows):
            colour = "white" if abs(grid[i, j]) > 0.6 * limit else "black"
            right.text(
                j,
                i,
                f"{grid[i, j] + 0.0:.2f}".replace("-0.00", "0.00"),
                ha="center",
                va="center",
                color=colour,
            )
            if (c, w) in time_matched:
                right.add_patch(
                    pf.plt.Rectangle(
                        (j - 0.5, i - 0.5),
                        1,
                        1,
                        fill=False,
                        edgecolor="black",
                        linewidth=1,
                    )
                )
    right.plot(
        [],
        [],
        marker="s",
        linestyle="",
        markerfacecolor="none",
        markeredgecolor="black",
        label="Same time after treatment",
    )
    right.legend(loc="lower left", bbox_to_anchor=(0, 1.0), frameon=False)
    right.set_xticks(range(len(windows)), [WINDOW_LABELS[w] for w in windows])
    right.set_yticks(range(len(COMPARISONS)), COMPARISONS)
    right.set_xlabel("This run's window")
    right.set_ylabel("Paper comparison")
    right.spines[["left", "bottom"]].set_visible(False)
    right.tick_params(length=0)
    fig.colorbar(
        image,
        ax=right,
        label="Spearman correlation, all genes tested in both",
        shrink=0.9,
    )

    pf.panel_label(left, "a")
    pf.panel_label(right, "b")
    suffix = "" if run == "joint" else f"_{run.replace('-', '_')}"
    pf.save(fig, f"figures/paper_gene_overlap{suffix}_domen2026_dataset_s1")


if __name__ == "__main__":
    main(*sys.argv[1:3])
