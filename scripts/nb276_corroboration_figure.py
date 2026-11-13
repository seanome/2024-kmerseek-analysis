#!/usr/bin/env python3
"""Figure 276q: which alphabets corroborate the segment, and which are blind to it.

Each alphabet is asked the same question on its own, with no averaging and no
combination: does the fixed segment P66 197-232 against CD47 109-144 score better
than the best that segment manages against any of 437 human proteins matched to
CD47 on length and acidic fraction?

  a  the answer in each alphabet's own units. The bar is the background's 5th to
     95th percentile, the dot its median, the diamond CD47. Scores are not
     comparable between alphabets, which is the point of showing them raw: in a
     two-letter alphabet CD47 scores 21 and still sits below the background.
  b  the same thing as one number per alphabet, against the uncorrected and
     Bonferroni-corrected thresholds, ordered by how many classes the alphabet
     has.

Run with the project interpreter:
  /Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3 \
      scripts/nb276_corroboration_figure.py
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

#: Colours carry the same meaning as in every other figure of this notebook.
C_CD47 = pf.OKABE_ITO["vermillion"]  # the measurement on CD47
C_BG = pf.GREY  # the matched-protein background
C_FINE = pf.OKABE_ITO["blue"]  # an alphabet that corroborates
#: Where the split in panel b falls, read off the result rather than chosen.
CLASS_SPLIT = 7


def main() -> None:
    pf.use_style()
    FIGDIR.mkdir(parents=True, exist_ok=True)

    summary = pl.read_csv(OUTDIR / "28_segment_corroboration_by_alphabet.csv")
    background = pl.read_csv(OUTDIR / "29_segment_corroboration_background.csv")
    summary = summary.sort("n_classes", "alphabet")
    names = summary["alphabet"].to_list()
    labels = [
        f"{a} ({n})" for a, n in zip(summary["alphabet"], summary["n_classes"])
    ]
    y = np.arange(len(names))
    n_background = int(summary["n_background"][0])
    corrected = 0.05 / summary.height

    fig, axes = pf.figure(pf.TWO_COLUMN_MM, 92, ncols=2, sharey=True)

    ax = axes[0]
    for i, name in enumerate(names):
        scores = background.filter(pl.col("alphabet") == name)["score"].to_numpy()
        low, high = np.percentile(scores, [5, 95])
        ax.plot([low, high], [i, i], color=C_BG, solid_capstyle="butt",
                label="background, 5th to 95th percentile" if i == 0 else None)
        ax.plot([np.median(scores)], [i], marker="o", color=C_BG, markersize=3,
                label="background median" if i == 0 else None)
    ax.scatter(summary["segment_score_on_cd47"], y, marker="D", s=14,
               color=C_CD47, edgecolor="black", zorder=4, label="CD47")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Segment score in that alphabet")
    pf.panel_label(ax, "a")

    ax = axes[1]
    labelled = set()
    for i, row in enumerate(summary.iter_rows(named=True)):
        shown = max(row["p_value"], 5e-4)
        corroborates = row["p_value"] < 0.05
        ax.plot([shown, 1.0], [i, i], color=C_BG, lw=0.3, zorder=1)
        key = "yes" if corroborates else "no"
        ax.scatter([shown], [i], s=16, zorder=3,
                   color=C_FINE if corroborates else C_BG,
                   edgecolor="black",
                   label=None if key in labelled else
                   ("corroborates at p < 0.05" if corroborates else "does not"))
        labelled.add(key)
    ax.axvline(0.05, color="black", ls=":", zorder=2, label="p = 0.05")
    ax.axvline(corrected, color=C_CD47, ls="--", zorder=2,
               label=f"p = {corrected:.4f}, Bonferroni for {summary.height} alphabets")
    ax.set_xscale("log")
    ax.set_xlim(3e-4, 1.6)
    ax.set_xticks([1e-3, 1e-2, 0.05, 1])
    ax.set_xticklabels(["0.001", "0.01", "0.05", "1"])
    ax.set_xlabel(
        f"Share of {n_background} length- and acid-matched\n"
        f"human proteins scoring at least as high"
    )
    # Rows run top to bottom in increasing class count, so the coarse alphabets are
    # above the line and the fine ones below it. The labels follow that, not the
    # arithmetic of the axis, which is inverted.
    first_fine = next(
        i for i, n in enumerate(summary["n_classes"]) if n >= CLASS_SPLIT
    )
    split = first_fine - 0.5
    ax.axhline(split, color="black", lw=0.3, zorder=1)
    ax.text(1.45, split - 0.35, f"fewer than {CLASS_SPLIT} classes", ha="right",
            va="bottom")
    ax.text(1.45, split + 0.35, f"{CLASS_SPLIT} classes and more", ha="right",
            va="top")
    pf.panel_label(ax, "b")

    pf.shared_legend(fig, ncol=3)
    pf.save(fig, FIGDIR / "276q_segment_corroboration_by_alphabet")

    n_yes = int((summary["p_value"] < 0.05).sum())
    n_bonf = int((summary["p_value"] < corrected).sum())
    fine = summary.filter(pl.col("n_classes") >= CLASS_SPLIT)
    coarse = summary.filter(pl.col("n_classes") < CLASS_SPLIT)
    print("276q", {
        "n_alphabets": summary.height,
        "n_corroborating_at_0.05": n_yes,
        "n_clearing_bonferroni": n_bonf,
        "bonferroni_threshold": round(corrected, 5),
        "fine_alphabets": fine.height,
        "fine_corroborating": int((fine["p_value"] < 0.05).sum()),
        "coarse_alphabets": coarse.height,
        "coarse_corroborating": int((coarse["p_value"] < 0.05).sum()),
        "coarse_p_range": f"{coarse['p_value'].min():.2f}-{coarse['p_value'].max():.2f}",
    })
    print(f"\nFigure in {FIGDIR}")


if __name__ == "__main__":
    main()
