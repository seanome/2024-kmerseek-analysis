#!/usr/bin/env python3
"""Compare the three fits against Domen et al. 2026: one model over all 52 libraries, each
window fitted alone, and the paper's own four comparisons fitted alone.

Usage: plot_three_fits.py   (run in this folder, after compare_paper_genes.py and
plot_paper_gene_overlap.py for all three runs)
Writes three_fits_vs_paper.tsv and figures/three_fits_vs_paper_domen2026_dataset_s1.{pdf,svg,png}.
"""

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "notebooks"))
import pubfig as pf  # noqa: E402

COMPARISONS = ["2 h", "24 h", "13-49 d", "132 d"]
# (folder, label, colour, marker)
RUNS = {
    "joint": (".", "One model, all 52 libraries, colony term", pf.GREY, "o"),
    "each-window": (
        "each-alone-comparison/each-window",
        "Each of 6 windows alone, 52 libraries",
        pf.OKABE_ITO["blue"],
        "s",
    ),
    "paper": (
        "each-alone-comparison/paper",
        "Paper's 4 comparisons alone, its 38 libraries",
        pf.OKABE_ITO["vermillion"],
        "^",
    ),
}


def load():
    rows = []
    for run, (folder, *_rest) in RUNS.items():
        folder = Path(folder)
        found = pl.read_csv(folder / "paper_gene_summary.tsv", separator="\t")
        agree = pl.read_csv(folder / "paper_gene_agreement.tsv", separator="\t")
        for r in found.filter(pl.col("comparison") != "all").iter_rows(named=True):
            hits = r["p_lt_0_05_same_direction"] + r["padj_lt_0_05_same_direction"]
            for a in agree.filter(pl.col("comparison") == r["comparison"]).iter_rows(
                named=True
            ):
                rows.append(
                    {
                        "run": run,
                        "comparison": r["comparison"],
                        "window": a["window"],
                        "paper_listed": r["listed"],
                        "listed_found_same_direction": hits,
                        "listed_found_pct": round(100 * hits / r["listed"], 1),
                        "genes_in_both": a["genes_in_both"],
                        "spearman": a["spearman"],
                        "paper_p_lt_0_05": a["paper_p_lt_0_05"],
                        "also_p_lt_0_05_here_same_direction": a[
                            "also_p_lt_0_05_here_same_direction"
                        ],
                    }
                )
    return pl.DataFrame(rows)


def main():
    table = load()
    table.write_csv("three_fits_vs_paper.tsv", separator="\t")
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=200):
        print(table)

    pf.use_style()
    fig, (left, right) = pf.figure(pf.TWO_COLUMN_MM, 60, ncols=2)
    y = np.arange(len(COMPARISONS))
    offsets = {"joint": -0.22, "each-window": 0.0, "paper": 0.22}
    for run, (_folder, label, colour, marker) in RUNS.items():
        t = table.filter(pl.col("run") == run)
        # A listed gene counts once per comparison, so one value per comparison.
        found = t.unique(subset="comparison", keep="first")
        for i, c in enumerate(COMPARISONS):
            v = found.filter(pl.col("comparison") == c)["listed_found_pct"][0]
            left.plot(v, i + offsets[run], marker=marker, color=colour, linestyle="")
            for rho in t.filter(pl.col("comparison") == c)["spearman"]:
                right.plot(
                    rho, i + offsets[run], marker=marker, color=colour, linestyle=""
                )
        left.plot([], [], marker=marker, color=colour, linestyle="", label=label)
    for ax in (left, right):
        ax.set_yticks(y, COMPARISONS)
        ax.invert_yaxis()
        ax.set_ylabel("Paper comparison")
        for i in y:
            ax.axhline(i, color="#DDDDDD", linewidth=0.3, zorder=0)
    left.set_xlim(-1, 40)
    left.set_xlabel("Paper's listed genes found here,\np < 0.05 same direction (%)")
    right.set_xlim(0, 0.6)
    right.set_xlabel("Spearman correlation with the paper,\nall genes tested in both")
    pf.shared_legend(fig, ncol=3)
    pf.panel_label(left, "a")
    pf.panel_label(right, "b")
    pf.save(fig, "figures/three_fits_vs_paper_domen2026_dataset_s1")


if __name__ == "__main__":
    main()
