#!/usr/bin/env python3
"""Figure 276r: what each alphabet gains over chance here, and which merges give it.

  a  matched columns observed against expected from that alphabet's own chance
     rate. Above the diagonal is signal; the coarse alphabets match far more
     columns and expect far more, so they sit on it.
  b  that excess on its own does not decide it. Excess against p-value, which
     shows the overlap: hp_kyte_doolittle2 beats chance by 6.4 columns and fails,
     wass14 beats it by 7.1 and passes, because a coarse alphabet's background is
     far wider. The two alphabets that get furthest are the two that accept an
     acidic residue against a basic one.

Run with the project interpreter:
  /Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3 \
      scripts/nb276_merges_figure.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "notebooks"))
import pubfig as pf  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nb276_p66_cd47_structure import OUTDIR  # noqa: E402

FIGDIR = Path(__file__).resolve().parent.parent / "figures"
C_YES = pf.OKABE_ITO["blue"]  # corroborates at p < 0.05
C_NO = pf.GREY
C_KE = pf.OKABE_ITO["vermillion"]  # accepts an acidic residue against a basic one


def main() -> None:
    import matplotlib.pyplot as plt

    pf.use_style()
    FIGDIR.mkdir(parents=True, exist_ok=True)
    merges = pl.read_csv(OUTDIR / "30_segment_merges_by_alphabet.csv").sort(
        "n_classes", "alphabet"
    )

    fig, axes = pf.figure(pf.TWO_COLUMN_MM, 80, ncols=2)

    ax = axes[0]
    limit = float(merges["columns_called_a_match"].max()) * 1.12
    ax.plot([0, limit], [0, limit], color="black", lw=0.5, ls=":",
            label="what chance gives")
    # Only the alphabets the text discusses are labelled; labelling all 19 puts
    # five names on top of each other at the bottom left.
    labelled = {"hsdm17", "sdm12", "funcgroups8", "protein20", "polarity4",
                "hp_kyte_doolittle2", "gbmr7"}
    for row in merges.iter_rows(named=True):
        ax.scatter([row["expected_by_chance"]], [row["columns_called_a_match"]],
                   s=18, zorder=3,
                   color=C_YES if row["corroborates_at_0_05"] else C_NO,
                   edgecolor="black")
        if row["alphabet"] in labelled:
            ax.annotate(row["alphabet"], (row["expected_by_chance"],
                                          row["columns_called_a_match"]),
                        textcoords="offset points", xytext=(4, 3), fontsize=4.4)
    ax.set_xlabel("Matched columns expected from that alphabet's chance rate")
    ax.set_ylabel("Matched columns observed (of 36)")
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    pf.panel_label(ax, "a")

    ax = axes[1]
    for row in merges.iter_rows(named=True):
        shown = max(row["p_value"], 5e-4)
        ax.scatter([row["excess_over_chance"]], [shown], s=18, zorder=3,
                   color=C_YES if row["corroborates_at_0_05"] else C_NO,
                   edgecolor="black")
        if row["of_those_acidic_against_basic"] and row["corroborates_at_0_05"]:
            ax.scatter([row["excess_over_chance"]], [shown], marker="*", s=30,
                       color=C_KE, zorder=4)
        if row["alphabet"] in labelled:
            ax.annotate(row["alphabet"], (row["excess_over_chance"], shown),
                        textcoords="offset points", xytext=(4, 3), fontsize=4.4)
    ax.axhline(0.05, color="black", lw=0.5, ls=":")
    ax.set_yscale("log")
    ax.set_xlabel("Matched columns above chance")
    ax.set_ylabel("p-value against the matched background")
    pf.panel_label(ax, "b")

    handles = [
        plt.Line2D([], [], marker="o", ls="", markersize=4, color=C_YES,
                   markeredgecolor="black", label="corroborates at p < 0.05"),
        plt.Line2D([], [], marker="o", ls="", markersize=4, color=C_NO,
                   markeredgecolor="black", label="does not"),
        plt.Line2D([], [], marker="*", ls="", markersize=6, color=C_KE,
                   label="corroborates and accepts K against E"),
        plt.Line2D([], [], ls=":", color="black", label="what chance gives"),
    ]
    fig.legend(handles=handles, loc="outside upper center", ncol=4)
    pf.save(fig, FIGDIR / "276r_which_merges_carry_the_segment")
    print("276r", {
        "n_alphabets": merges.height,
        "corroborating": int(merges["corroborates_at_0_05"].sum()),
        "with_acidic_basic_merge": int(
            (merges["of_those_acidic_against_basic"] > 0).sum()
        ),
        "funcgroups8_excess": float(
            merges.filter(pl.col("alphabet") == "funcgroups8")["excess_over_chance"][0]
        ),
        "hsdm17_excess": float(
            merges.filter(pl.col("alphabet") == "hsdm17")["excess_over_chance"][0]
        ),
    })


if __name__ == "__main__":
    main()
