#!/usr/bin/env python3
"""Figures for the seed-and-extend part of notebook 276.

  276j  the best seed-and-extend segment written out residue by residue, with the
        P66 loop and CD47's SIRPalpha face marked.
  276k  each candidate's score against its two null distributions, with the
        uncorrected and Bonferroni-corrected thresholds.

Reads only the tables the seed-and-extend scripts wrote.

Run with the project interpreter:
  /Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3 \
      scripts/nb276_seed_extend_figures.py
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
C_LOOP = pubfig.OKABE_ITO["vermillion"]
C_FACE = pubfig.OKABE_ITO["blue"]
C_BG = pubfig.GREY
C_MATCH = pubfig.OKABE_ITO["bluish_green"]
#: The sweep's size, for the corrected threshold.
N_SETTINGS = 132


def figure_alignment() -> dict:
    """The strongest seed-and-extend segment, printed as the two sequences."""
    import matplotlib.pyplot as plt

    sweep = pl.read_csv(OUTDIR / "11_seed_extend_loop.csv").filter(
        pl.col("reaches_loop")
    )
    overlap = pl.read_csv(OUTDIR / "13_seed_extend_face_overlap.csv")
    follow = pl.read_csv(OUTDIR / "12_seed_extend_followup.csv")
    # The best candidate: lowest p-value in the focused test.
    best = follow.sort("p_composition_null").row(0, named=True)
    row = sweep.filter(
        (pl.col("alphabet") == best["alphabet"])
        & (pl.col("seed_length") == best["seed_length"])
        & (pl.col("penalty") == best["penalty"])
    ).row(0, named=True)
    face_row = overlap.filter(
        (pl.col("alphabet") == best["alphabet"])
        & (pl.col("seed_length") == best["seed_length"])
        & (pl.col("penalty") == best["penalty"])
    ).row(0, named=True)

    loop = pl.read_csv(OUTDIR / "01_loop_numbering.csv").row(0, named=True)
    face_positions = set(
        pl.read_csv(OUTDIR / "04_cd47_sirpalpha_face.csv")["cd47_uniprot_position"]
        .to_list()
    )
    p66_start = int(row["p66_span"].split("-")[0])
    cd47_start = int(row["cd47_span"].split("-")[0])
    p66_res, cd47_res = row["p66_residues"], row["cd47_residues"]

    per_line = 40
    n_lines = int(np.ceil(len(p66_res) / per_line))
    fig, ax = pubfig.figure(
        width_mm=pubfig.TWO_COLUMN_MM, height_mm=20 + 24 * n_lines
    )
    y = 0.0
    for line in range(n_lines):
        lo = line * per_line
        hi = min(lo + per_line, len(p66_res))
        for col, i in enumerate(range(lo, hi)):
            p66_pos, cd47_pos = p66_start + i, cd47_start + i
            in_loop = loop["full_start"] <= p66_pos <= loop["full_end"]
            in_face = cd47_pos in face_positions
            same = p66_res[i] == cd47_res[i]
            ax.text(col, y, p66_res[i], ha="center", va="center", fontsize=5.5,
                    color=C_LOOP if in_loop else "black",
                    fontweight="bold" if in_loop else "normal")
            ax.text(col, y - 0.55, "|" if same else " ", ha="center", va="center",
                    fontsize=5.5, color=C_MATCH)
            ax.text(col, y - 1.1, cd47_res[i], ha="center", va="center", fontsize=5.5,
                    color=C_FACE if in_face else "black",
                    fontweight="bold" if in_face else "normal")
        ax.text(-1.2, y, f"P66 {p66_start + lo}", ha="right", va="center", fontsize=4.6)
        ax.text(-1.2, y - 1.1, f"CD47 {cd47_start + lo}", ha="right", va="center",
                fontsize=4.6)
        y -= 2.6
    handles = [
        plt.Line2D([], [], marker="s", ls="", ms=4, color=C_LOOP,
                   label=f"P66 loop in the claim ({loop['full_start']}-{loop['full_end']})"),
        plt.Line2D([], [], marker="s", ls="", ms=4, color=C_FACE,
                   label=f"CD47 residue that touches SIRPalpha "
                         f"({face_row['n_face_residues_touched']} in this segment)"),
        plt.Line2D([], [], marker="|", ls="", ms=6, color=C_MATCH,
                   label="identical residue"),
    ]
    fig.legend(handles=handles, loc="outside upper center", ncol=3, fontsize=5)
    ax.set_xlim(-9, per_line)
    ax.set_ylim(y + 1.4, 1.3)
    ax.axis("off")
    ax.set_title(
        f"{best['alphabet']} ({best['n_classes']} classes), seed "
        f"{best['seed_length']}, mismatch penalty {best['penalty']}: score "
        f"{best['score']} over {best['segment_length']} residues, "
        f"p = {best['p_composition_null']:.4f} against shuffled CD47\n"
        f"Bonferroni threshold for the {N_SETTINGS}-setting sweep is "
        f"p < {0.05 / N_SETTINGS:.5f}, so this is a lead and not a result",
        fontsize=5, pad=3,
    )
    pubfig.save(fig, FIGDIR / "276j_seed_extend_best_alignment")
    return {
        "alphabet": best["alphabet"],
        "p66_span": row["p66_span"],
        "cd47_span": row["cd47_span"],
        "face_residues": face_row["face_residues_touched"],
        "fraction_in_ectodomain": face_row["fraction_in_ectodomain"],
        "p_composition": best["p_composition_null"],
        "p_neighbour_pairs": best["p_neighbour_pairs_null"],
    }


def figure_nulls() -> dict:
    """Each candidate's score against its nulls, and the corrected threshold."""
    follow = pl.read_csv(OUTDIR / "12_seed_extend_followup.csv").sort(
        "p_composition_null"
    )
    corrected = 0.05 / N_SETTINGS

    fig, ax = pubfig.figure(width_mm=pubfig.ONE_COLUMN_MM, height_mm=58)
    x = np.arange(follow.height)
    ax.scatter(x - 0.12, follow["p_composition_null"], s=16, color=C_BG,
               edgecolor="black", linewidth=0.3, zorder=3,
               label="against shuffled CD47 (composition kept)")
    ax.scatter(x + 0.12, follow["p_neighbour_pairs_null"], s=16, marker="^",
               color=C_FACE, edgecolor="black", linewidth=0.3, zorder=3,
               label="against shuffled CD47 (neighbouring pairs kept)")
    ax.axhline(0.05, color="black", lw=0.5, ls=":", zorder=1,
               label="p = 0.05, uncorrected")
    ax.axhline(corrected, color=C_LOOP, lw=0.7, ls="--", zorder=1,
               label=f"p = {corrected:.5f}, Bonferroni for {N_SETTINGS} settings")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{a}\nseed {s}" for a, s in
         zip(follow["alphabet"], follow["seed_length"])], fontsize=4.6,
    )
    ax.set_ylabel("p-value from 5_000 shuffles\n(lower is stronger)")
    pubfig.shared_legend(fig, ncol=1, fontsize=4.6)
    pubfig.save(fig, FIGDIR / "276k_seed_extend_null_tests")
    return {
        "corrected_threshold": corrected,
        "n_candidates": follow.height,
        "n_below_uncorrected": int((follow["p_composition_null"] < 0.05).sum()),
        "n_below_corrected_both": int(
            follow.filter(
                (pl.col("p_composition_null") < corrected)
                & (pl.col("p_neighbour_pairs_null") < corrected)
            ).height
        ),
    }


def main() -> None:
    pubfig.use_style()
    FIGDIR.mkdir(parents=True, exist_ok=True)
    print("276j", figure_alignment())
    print("276k", figure_nulls())
    print(f"\nFigures in {FIGDIR}")


if __name__ == "__main__":
    main()
