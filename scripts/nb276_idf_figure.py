#!/usr/bin/env python3
"""Figure 276s: what an IDF weighting does to the segment and to the loop.

  a  the segment's summed IDF on CD47 against the matched background's, per
     alphabet. In the fine alphabets both are zero, because the segment contains
     no exactly matching 5-mer at all: its similarity is scattered single
     residues, not runs. In the coarse ones 5-mers do match and their IDF is near
     zero, because they are the commonest 5-mers there are.
  b  how rare the loop's own 5-mers are in the human proteome, against the rest
     of the segment and against the maximum a 5-mer can reach. The loop's k-mers
     are rare in the fine alphabets and common in the coarse ones, and in every
     alphabet but the two-letter one none of them occurs in CD47.

Run with the project interpreter:
  /Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3 \
      scripts/nb276_idf_figure.py
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
C_CD47 = pf.OKABE_ITO["vermillion"]
C_BG = pf.GREY
C_LOOP = pf.OKABE_ITO["bluish_green"]
C_FLANK = pf.OKABE_ITO["blue"]


def main() -> None:
    import matplotlib.pyplot as plt

    pf.use_style()
    FIGDIR.mkdir(parents=True, exist_ok=True)
    seg = pl.read_csv(OUTDIR / "32_idf_weighted_segment.csv").sort("n_classes")
    loops = pl.read_csv(OUTDIR / "33_loop_kmer_idf.csv").sort("n_classes")
    labels = [f"{a}\n({n})" for a, n in zip(seg["alphabet"], seg["n_classes"])]
    x = np.arange(seg.height)

    fig, axes = pf.figure(pf.TWO_COLUMN_MM, 66, ncols=2)

    ax = axes[0]
    width = 0.36
    ax.bar(x - width / 2, seg["summed_idf_on_cd47"], width, color=C_CD47,
           label="CD47")
    ax.bar(x + width / 2, seg["background_max_summed_idf"], width, color=C_BG,
           label="best of the 437 matched proteins")
    for i, row in enumerate(seg.iter_rows(named=True)):
        if row["n_matched_kmers_in_segment"] == 0:
            ax.annotate("no 5-mer\nmatches at all", (i - width / 2, 0.3),
                        ha="center", va="bottom", fontsize=4.2, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Summed IDF of the segment's matching 5-mers")
    pf.panel_label(ax, "a")

    ax = axes[1]
    ax.bar(x - width / 2, loops["loop_median_idf"], width, color=C_LOOP,
           label="the loop's 5-mers")
    ax.bar(x + width / 2, loops["flank_median_idf"], width, color=C_FLANK,
           label="the rest of the segment's 5-mers")
    ax.axhline(float(loops["max_possible_idf"][0]), color="black", ls=":",
               label="a 5-mer unique in the proteome")
    for i, row in enumerate(loops.iter_rows(named=True)):
        ax.annotate(f"{row['loop_kmers_absent_from_cd47']}/{row['loop_kmers']}",
                    (i - width / 2, row["loop_median_idf"]), ha="center",
                    va="bottom", fontsize=4.2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Median IDF of a 5-mer (higher is rarer)")
    pf.panel_label(ax, "b")
    ax.set_title("Numbers above the bars: loop 5-mers absent from CD47",
                 fontsize=5, pad=2)

    pf.shared_legend(fig, ncol=3)
    pf.save(fig, FIGDIR / "276s_idf_weighting")
    print("276s", {
        "alphabets": seg.height,
        "with_zero_matching_kmers": int(
            (seg["n_matched_kmers_in_segment"] == 0).sum()
        ),
        "idf_weighted_p_values": seg["idf_weighted_p_value"].to_list(),
        "loop_median_idf_protein20": float(
            loops.filter(pl.col("alphabet") == "protein20")["loop_median_idf"][0]
        ),
        "loop_median_idf_hp": float(
            loops.filter(pl.col("alphabet") == "hp_pbotc_1st_ed2")["loop_median_idf"][0]
        ),
    })


if __name__ == "__main__":
    main()
