#!/usr/bin/env python3
"""Plot the size-factor test: each of the paper's four comparisons fitted with only the size
factors changed, on this run's counts and on the paper's counts.

Usage: plot_size_factors.py   (run in this folder; reads size_factors_fits.tsv, which
each-alone-checks/size_factors.R writes)
Writes size_factors_summary.tsv (listed genes found over all four comparisons, per fit) and
figures/size_factor_test_domen2026_dataset_s1.{svg,png}.
"""

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "notebooks"))
import pubfig as pf  # noqa: E402

COMPARISONS = ["2 h", "24 h", "13-49 d", "132 d"]
COUNTS = {"this_run": "This run's counts", "paper": "The paper's counts (Dataset S1)"}
# size factor kind -> (label, colour, marker, row offset)
KINDS = {
    "default": (
        "DESeq2 default, from the same counts",
        pf.OKABE_ITO["blue"],
        "o",
        -0.3,
    ),
    "swapped": (
        "DESeq2 default, from the other data set's counts",
        pf.OKABE_ITO["vermillion"],
        "D",
        -0.15,
    ),
    "poscounts": ("poscounts, from the same counts", pf.OKABE_ITO["bluish_green"], "s", 0.0),
    "total": ("Library total of the same counts", pf.OKABE_ITO["orange"], "^", 0.15),
    # Only on this run's counts: the pipeline's gene-length correction left out.
    "default_no_gene_lengths": (
        "DESeq2 default, from the same counts, no gene-length correction (this run only)",
        pf.OKABE_ITO["reddish_purple"],
        "v",
        0.3,
    ),
}
FOUND = "listed_p_lt_0_05_same_direction"


def summary(fits):
    return (
        fits.group_by("counts", "size_factors", maintain_order=True)
        .agg(
            paper_listed=pl.col("paper_listed").sum(),
            listed_tested=pl.col("listed_tested").sum(),
            listed_found_same_direction=pl.col(FOUND).sum(),
            p_le_0_05_2fold_up=pl.col("p_le_0_05_2fold_up").sum(),
            p_le_0_05_2fold_down=pl.col("p_le_0_05_2fold_down").sum(),
        )
        .with_columns(
            listed_found_pct=(
                100 * pl.col("listed_found_same_direction") / pl.col("paper_listed")
            ).round(1)
        )
    )


def main():
    fits = pl.read_csv("size_factors_fits.tsv", separator="\t")
    assert fits.height == len(COMPARISONS) * (2 * len(KINDS) - 1), fits.height
    total = summary(fits)
    total.write_csv("size_factors_summary.tsv", separator="\t")
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=200):
        print(total)

    pf.use_style()
    fig, axs = pf.figure(pf.TWO_COLUMN_MM, 105, nrows=2, ncols=2, sharey=True, sharex="row")
    y = np.arange(len(COMPARISONS))
    drawn = 0
    for col, (counts, title) in enumerate(COUNTS.items()):
        top, bottom = axs[0, col], axs[1, col]
        for kind, (label, colour, marker, offset) in KINDS.items():
            t = fits.filter(
                (pl.col("counts") == counts) & (pl.col("size_factors") == kind)
            )
            if t.is_empty():
                continue
            for i, c in enumerate(COMPARISONS):
                (r,) = t.filter(pl.col("comparison") == c).iter_rows(named=True)
                style = dict(marker=marker, color=colour, linestyle="")
                top.plot(100 * r[FOUND] / r["paper_listed"], i + offset, **style)
                bottom.plot(r["p_le_0_05_2fold_up"], i + offset, **style)
                bottom.plot(-r["p_le_0_05_2fold_down"], i + offset, **style)
                drawn += 3
            axs[0, 0].plot([], [], marker=marker, color=colour, linestyle="", label=label)
        top.set_title(title)
        top.set_xlabel("Paper's listed genes found,\np < 0.05 in the listed direction (%)")
        top.set_xlim(0, 90)
        bottom.axvline(0, color="black", linewidth=0.5)
        bottom.set_xlabel(
            "Genes with p ≤ 0.05 and more than 2-fold change (n)\n"
            "lower in treated  |  higher in treated"
        )
        # Counts to the left of zero are genes going down; show them without a minus sign.
        bottom.xaxis.set_major_formatter(lambda v, _pos: f"{abs(v):g}")
        for ax in (top, bottom):
            for i in y:
                ax.axhline(i, color="#DDDDDD", linewidth=0.3, zorder=0)
    assert drawn == 3 * fits.height
    for ax in axs[:, 0]:
        ax.set_yticks(y, COMPARISONS)
        ax.set_ylabel("Paper comparison")
    axs[0, 0].invert_yaxis()
    pf.shared_legend(fig, ncol=2, title="Size factors used in the fit")
    for ax, letter in zip(axs.flat, "abcd"):
        pf.panel_label(ax, letter)
    pf.save(fig, "figures/size_factor_test_domen2026_dataset_s1")


if __name__ == "__main__":
    main()
