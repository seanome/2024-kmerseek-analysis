#!/usr/bin/env python3
"""Figures for what the 36-residue segment is, and for combining alphabets.

  276n  the segment's specificity: every 36-residue window of P66 scored against
        CD47, and P66 197-232 against 239 immunoglobulin domains.
  276o  combining alphabets against the best single alphabet, on the whole
        segment and on the loop's own columns.

Run with the project interpreter:
  /Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3 \
      scripts/nb276_segment_and_combination_figures.py
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
C_SEG = pubfig.OKABE_ITO["vermillion"]
C_BG = pubfig.GREY
C_FACE = pubfig.OKABE_ITO["blue"]
C_OTHER = pubfig.OKABE_ITO["bluish_green"]
#: The segment, from the prespecified test.
P66_SEGMENT = (197, 232)


def figure_segment() -> dict:
    import matplotlib.pyplot as plt

    windows = pl.read_csv(OUTDIR / "23_p66_windows_vs_cd47.csv")
    ig = pl.read_csv(OUTDIR / "22_segment_vs_ig_domains.csv")
    loop = pl.read_csv(OUTDIR / "01_loop_numbering.csv").row(0, named=True)
    observed = windows.filter(
        pl.col("p66_start") == P66_SEGMENT[0]
    )["best_score"][0]

    fig, axes = pubfig.figure(width_mm=pubfig.TWO_COLUMN_MM, height_mm=58, ncols=2)
    ax = axes[0]
    ax.plot(windows["p66_start"], windows["best_score"], color=C_BG, lw=0.6,
            label="every 36-residue window of P66")
    top = windows.filter(pl.col("best_score") == windows["best_score"].max())
    ax.scatter(top["p66_start"], top["best_score"], s=3, color=C_OTHER, zorder=3,
               label=f"windows at the top score ({top.height})")
    ax.axvspan(loop["full_start"], loop["full_end"], color=C_SEG, alpha=0.3, lw=0,
               label=f"the loop ({loop['full_start']}-{loop['full_end']})")
    ax.scatter([P66_SEGMENT[0]], [observed], s=28, marker="D", color=C_SEG,
               edgecolor="black", linewidth=0.3, zorder=4,
               label=f"the window at {P66_SEGMENT[0]}")
    ax.set_xlabel("start of the 36-residue window in P66")
    ax.set_ylabel("best score against CD47")
    pubfig.panel_label(ax, "a")
    ax.legend(fontsize=4.4, loc="lower center", ncol=2)
    ax.set_title(
        "Two regions of P66 reach the top score against CD47:\n"
        "197-264 and 517-600. Everything else is 4.5 or less.",
        fontsize=5, pad=2,
    )

    ax = axes[1]
    bins = np.arange(-0.25, max(ig["best_score"].max(), observed) + 0.75, 0.5)
    ax.hist(ig["best_score"], bins=bins, color=C_BG, edgecolor="white",
            linewidth=0.3, label=f"{ig.height} SCOPe b.1.1 domains")
    ax.axvline(observed, color=C_SEG, lw=1.2, label=f"CD47 ({observed})")
    ax.set_xlabel("best score for P66 197-232 against the target")
    ax.set_ylabel("immunoglobulin domains")
    pubfig.panel_label(ax, "b")
    ax.legend(fontsize=4.6, loc="upper right")
    n_ge = int((ig["best_score"] >= observed).sum())
    ax.set_title(
        f"It is CD47, not immunoglobulins in general:\n"
        f"{n_ge} of {ig.height} domains reach CD47's score, median "
        f"{ig['best_score'].median()}",
        fontsize=5, pad=2,
    )
    pubfig.save(fig, FIGDIR / "276n_segment_specificity")
    return {
        "observed": observed,
        "n_windows": windows.height,
        "n_windows_at_top": int(
            (windows["best_score"] == windows["best_score"].max()).sum()
        ),
        "n_ig_reaching": n_ge,
        "ig_median": float(ig["best_score"].median()),
    }


def figure_combinations() -> dict:
    import matplotlib.pyplot as plt

    combined = pl.read_csv(OUTDIR / "25_combined_alphabets.csv")
    single = pl.read_csv(OUTDIR / "18_hsdm17_loop_vs_flanks_result.csv")
    single_loop = single.filter(
        pl.col("statistic") == "loop columns only"
    )["observed"][0]
    prespec = pl.read_csv(OUTDIR / "16_hsdm17_prespecified_result.csv").row(
        0, named=True
    )

    reached = combined.filter(pl.col("reaches_loop"))
    labels = [
        f"{row['combination']}\n{row['seeding_rule']} seed {row['seed_length']}"
        for row in reached.iter_rows(named=True)
    ] + ["hsdm17 alone\nseed 4 (the single best)"]
    p_total = reached["p_total"].to_list() + [prespec["p_primary"]]
    p_loop = reached["p_loop_among_reaching"].to_list() + [1.0]
    x = np.arange(len(labels))

    fig, ax = pubfig.figure(width_mm=pubfig.TWO_COLUMN_MM, height_mm=62)
    width = 0.38
    ax.bar(x - width / 2, [max(v, 0.002) for v in p_total], width, color=C_FACE,
           label="whole segment: share of 437 matched proteins reaching it")
    ax.bar(x + width / 2, [max(v, 0.002) for v in p_loop], width, color=C_SEG,
           label="the loop's own columns: share of those reaching the loop that beat it")
    ax.axhline(0.05, color="black", lw=0.5, ls=":", label="0.05")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=4.4)
    ax.set_ylabel("share of the matched background\n(lower is better on the left bar)")
    n_missing = combined.filter(~pl.col("reaches_loop")).height
    ax.set_title(
        f"Combining alphabets does worse than the best single one. "
        f"{n_missing} of {combined.height} combination settings never reach the loop "
        f"at all;\nthe ones that do are beaten by most of the background, and on the "
        f"loop's own columns every scheme including hsdm17 is beaten by all of it.",
        fontsize=5, pad=3,
    )
    pubfig.shared_legend(fig, ncol=1, fontsize=4.6)
    pubfig.save(fig, FIGDIR / "276o_combined_alphabets")
    return {
        "n_settings": combined.height,
        "n_never_reaching": n_missing,
        "best_combination_p_total": float(reached["p_total"].min())
        if not reached.is_empty()
        else None,
        "single_alphabet_p_total": prespec["p_primary"],
        "single_alphabet_loop_score": single_loop,
    }


def main() -> None:
    pubfig.use_style()
    FIGDIR.mkdir(parents=True, exist_ok=True)
    print("276n", figure_segment())
    print("276o", figure_combinations())
    print(f"\nFigures in {FIGDIR}")


if __name__ == "__main__":
    main()
