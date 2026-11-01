#!/usr/bin/env python3
"""Figure 276p: why averaging alphabets hurt, and where corroboration does happen.

Three panels, each one of the measurements in
scripts/nb276_why_combining_hurts.py:

  a  how often each alphabet calls a random pair of residues a match, against
     its number of classes. This is the scale the average is taken over.
  b  the 36 columns of the segment, split into the ones hsdm17 calls a match and
     the ones it calls a mismatch, scored both ways. The matches survive
     averaging; the mismatches get worse.
  c  each alphabet asked on its own whether the segment beats its own matched
     background. Corroboration is real, and it comes from the fine alphabets.

Run with the project interpreter:
  /Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3 \
      scripts/nb276_why_combining_figure.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "notebooks"))
import pubfig  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nb276_p66_cd47_structure import OUTDIR  # noqa: E402

FIGDIR = Path(__file__).resolve().parent.parent / "figures"
C_SINGLE = pubfig.OKABE_ITO["blue"]
C_AVG = pubfig.OKABE_ITO["vermillion"]
C_BG = pubfig.GREY
C_YES = pubfig.OKABE_ITO["bluish_green"]


def main() -> None:
    import matplotlib.pyplot as plt

    pubfig.use_style()
    FIGDIR.mkdir(parents=True, exist_ok=True)
    coll = pl.read_csv(OUTDIR / "26_alphabet_collision_rates.csv")
    cols = pl.read_csv(OUTDIR / "27_segment_columns_one_vs_average.csv")
    corr = pl.read_csv(OUTDIR / "28_segment_corroboration_by_alphabet.csv")

    fig, axes = pubfig.figure(
        width_mm=pubfig.TWO_COLUMN_MM, height_mm=58, ncols=3
    )

    ax = axes[0]
    ax.scatter(coll["n_classes"], coll["chance_two_residues_share_a_class"],
               s=14, color=C_BG, edgecolor="black", linewidth=0.3)
    mean = float(coll["chance_two_residues_share_a_class"].mean())
    ax.axhline(mean, color=C_AVG, lw=0.8, ls="--",
               label=f"panel mean {mean:.2f}")
    ax.set_xlabel("classes in the alphabet")
    ax.set_ylabel("chance two random residues\nare called a match")
    ax.set_ylim(0, 0.62)
    ax.legend(fontsize=4.6, loc="upper right")
    pubfig.panel_label(ax, "a")
    ax.set_title("A 2-class alphabet calls half of all pairs\na match; "
                 "protein20 calls 1 in 20.", fontsize=5, pad=2)

    ax = axes[1]
    matched = cols.filter(pl.col("hsdm17_column_score") > 0)
    mismatched = cols.filter(pl.col("hsdm17_column_score") < 0)
    groups = [
        (f"the {matched.height} columns\nhsdm17 calls a match",
         float(matched["hsdm17_column_score"].sum()),
         float(matched["averaged_column_score"].sum())),
        (f"the {mismatched.height} columns\nhsdm17 calls a mismatch",
         float(mismatched["hsdm17_column_score"].sum()),
         float(mismatched["averaged_column_score"].sum())),
        ("the whole 36-residue\nsegment",
         float(cols["hsdm17_column_score"].sum()),
         float(cols["averaged_column_score"].sum())),
    ]
    x = np.arange(len(groups))
    width = 0.36
    ax.bar(x - width / 2, [g[1] for g in groups], width, color=C_SINGLE,
           label="scored in hsdm17 alone")
    ax.bar(x + width / 2, [g[2] for g in groups], width, color=C_AVG,
           label="scored by averaging 19 alphabets")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([g[0] for g in groups], fontsize=4.4)
    ax.set_ylabel("total score")
    ax.legend(fontsize=4.6, loc="lower left")
    pubfig.panel_label(ax, "b")
    ax.set_title("The matches survive averaging. The mismatches\n"
                 "get worse, and there are more of them.", fontsize=5, pad=2)

    ax = axes[2]
    for row in corr.iter_rows(named=True):
        ax.scatter([row["n_classes"]], [max(row["p_value"], 1e-3)], s=14,
                   color=C_YES if row["corroborates_at_0_05"] else C_BG,
                   edgecolor="black", linewidth=0.3, zorder=3)
    ax.axhline(0.05, color="black", lw=0.5, ls=":", label="p = 0.05")
    ax.axhline(0.05 / corr.height, color=C_AVG, lw=0.7, ls="--",
               label=f"Bonferroni over {corr.height} alphabets")
    ax.set_yscale("log")
    ax.set_xlabel("classes in the alphabet")
    ax.set_ylabel("p-value for the segment against\nits own matched background")
    ax.legend(fontsize=4.4, loc="lower left")
    pubfig.panel_label(ax, "c")
    n_yes = int(corr["corroborates_at_0_05"].sum())
    ax.set_title(f"Corroboration is real and comes from the fine\nalphabets: "
                 f"{n_yes} of {corr.height} at p < 0.05, none below 7 classes.",
                 fontsize=5, pad=2)

    pubfig.save(fig, FIGDIR / "276p_why_combining_alphabets_hurts")
    print("276p", {
        "panel_mean_collision": mean,
        "segment_hsdm17": float(cols["hsdm17_column_score"].sum()),
        "segment_averaged": float(cols["averaged_column_score"].sum()),
        "n_corroborating": n_yes,
        "n_alphabets": corr.height,
    })
    print(f"\nFigure in {FIGDIR}")


if __name__ == "__main__":
    main()
