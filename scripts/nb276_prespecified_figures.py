#!/usr/bin/env python3
"""Figures for the prespecified hsdm17 test and the all-alphabets summary.

  276l  CD47's score against 437 length- and acid-matched human proteins, and the
        same split into the loop's own columns and the flanking columns.
  276m  every alphabet in one picture: whether exact k-mers and seed-and-extend
        reach the loop, and how strong it is against shuffled CD47.

Run with the project interpreter:
  /Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3 \
      scripts/nb276_prespecified_figures.py
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
C_CD47 = pubfig.OKABE_ITO["vermillion"]
C_BG = pubfig.GREY
C_FACE = pubfig.OKABE_ITO["blue"]
C_LOOP = pubfig.OKABE_ITO["bluish_green"]
#: Bonferroni threshold the earlier 132-setting sweep is judged at.
SWEEP_THRESHOLD = 0.05 / 132


def figure_prespecified() -> dict:
    import matplotlib.pyplot as plt

    result = pl.read_csv(OUTDIR / "16_hsdm17_prespecified_result.csv").row(
        0, named=True
    )
    background = pl.read_csv(OUTDIR / "15_hsdm17_prespecified_background.csv")
    split = pl.read_csv(OUTDIR / "17_hsdm17_loop_vs_flanks_background.csv")
    split_result = pl.read_csv(OUTDIR / "18_hsdm17_loop_vs_flanks_result.csv")
    loop_obs = split_result.filter(
        pl.col("statistic") == "loop columns only"
    )["observed"][0]
    flank_obs = split_result.filter(
        pl.col("statistic") == "flanking columns only"
    )["observed"][0]

    fig, axes = pubfig.figure(
        width_mm=pubfig.TWO_COLUMN_MM, height_mm=62, ncols=2
    )
    ax = axes[0]
    scores = background["score"].to_list()
    bins = np.arange(-0.25, max(max(scores), result["observed_score"]) + 0.75, 0.5)
    ax.hist(scores, bins=bins, color=C_BG, edgecolor="white", linewidth=0.3,
            label=f"{background.height} matched human proteins")
    ax.axvline(result["observed_score"], color=C_CD47, lw=1.2,
               label=f"CD47 ({result['observed_score']})")
    ax.set_xlabel("score of the best segment covering the loop")
    ax.set_ylabel("matched human proteins")
    ax.set_yscale("log")
    pubfig.panel_label(ax, "a")
    ax.legend(fontsize=4.6, loc="upper right")
    ax.set_title(
        f"Prespecified: hsdm17, seed 4, penalty 0.5\n"
        f"{result['n_background_score_ge_observed']} of {background.height} reach "
        f"CD47's score, p = {result['p_primary']:.3f}",
        fontsize=5, pad=2,
    )

    ax = axes[1]
    reached = split.filter(pl.col("reaches_loop"))
    jitter = np.random.default_rng(276).uniform(-0.08, 0.08, size=reached.height)
    ax.scatter(np.zeros(reached.height) + jitter, reached["loop_score"], s=10,
               color=C_LOOP, edgecolor="none", label="the loop's own 7 columns")
    ax.scatter(np.ones(reached.height) + jitter, reached["flank_score"], s=10,
               color=C_FACE, edgecolor="none", label="the 29 flanking columns")
    ax.scatter([0], [loop_obs], s=34, marker="D", color=C_CD47,
               edgecolor="black", linewidth=0.3, zorder=4, label="CD47")
    ax.scatter([1], [flank_obs], s=34, marker="D", color=C_CD47,
               edgecolor="black", linewidth=0.3, zorder=4)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["loop columns", "flanking columns"], fontsize=5)
    ax.set_ylabel("score contributed")
    ax.set_xlim(-0.4, 1.4)
    pubfig.panel_label(ax, "b")
    ax.legend(fontsize=4.6, loc="upper left")
    ax.set_title(
        f"Where CD47's score comes from, against the "
        f"{reached.height} matched proteins\nthat also reach the loop. "
        f"On the loop, CD47 is the lowest of the {reached.height + 1}.",
        fontsize=5, pad=2,
    )
    pubfig.save(fig, FIGDIR / "276l_hsdm17_prespecified_test")
    return {
        "p_primary": result["p_primary"],
        "observed": result["observed_score"],
        "n_background": background.height,
        "loop_observed": loop_obs,
        "loop_background_median": float(reached["loop_score"].median()),
        "flank_observed": flank_obs,
        "flank_background_median": float(reached["flank_score"].median()),
    }


def figure_all_alphabets() -> dict:
    import matplotlib.pyplot as plt

    table = pl.read_csv(OUTDIR / "20_all_alphabets_summary.csv").sort(
        "n_classes", "alphabet"
    )
    names = [
        f"{a} ({n})" for a, n in zip(table["alphabet"], table["n_classes"])
    ]
    y = np.arange(table.height)

    fig, axes = pubfig.figure(
        width_mm=pubfig.TWO_COLUMN_MM, height_mm=96, ncols=2, sharey=True
    )
    ax = axes[0]
    exact = table["loop_kmers_in_cd47"].to_list()
    ax.barh(y, [max(v, 0.3) for v in exact], color=[
        C_LOOP if u else C_FACE for u in table["loop_is_one_repeated_class"]
    ], height=0.6)
    for i, v in enumerate(exact):
        ax.text(max(v, 0.3) * 1.15, i, str(v), va="center", fontsize=4.4)
    ax.set_xscale("log")
    ax.set_xlabel("exact k-mers covering the loop that occur in CD47\n(k 5 to 25, "
                  "0 drawn as a stub)")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=4.8)
    ax.invert_yaxis()
    pubfig.panel_label(ax, "a")

    ax = axes[1]
    p = table["seed_extend_best_p"].to_list()
    reaching = table["seed_extend_settings_reaching_loop"].to_list()
    for i, (value, n) in enumerate(zip(p, reaching)):
        if value is None or n == 0:
            ax.text(0.5, i, "never reaches the loop", va="center", ha="center",
                    fontsize=4.4, color=C_BG)
            continue
        shown = max(value, 1e-4)
        ax.scatter([shown], [i], s=18,
                   color=C_LOOP if table["loop_is_one_repeated_class"][i] else C_FACE,
                   edgecolor="black", linewidth=0.3, zorder=3)
        ax.text(shown * 1.4, i, f"{value:.2f}" if value >= 0.01 else "<0.01",
                va="center", fontsize=4.4)
    ax.axvline(0.05, color="black", lw=0.5, ls=":", label="p = 0.05, uncorrected")
    ax.axvline(SWEEP_THRESHOLD, color=C_CD47, lw=0.7, ls="--",
               label=f"p = {SWEEP_THRESHOLD:.5f}, Bonferroni for the sweep")
    ax.set_xscale("log")
    ax.set_xlim(5e-5, 2)
    ax.set_xlabel("best seed-and-extend p-value against shuffled CD47\n(lower is "
                  "stronger)")
    ax.legend(fontsize=4.4, loc="lower right")
    pubfig.panel_label(ax, "b")

    handles = [
        plt.Line2D([], [], marker="s", ls="", ms=5, color=C_LOOP,
                   label="alphabet writes the loop as one repeated class"),
        plt.Line2D([], [], marker="s", ls="", ms=5, color=C_FACE,
                   label="alphabet writes the loop as a varied string"),
    ]
    fig.legend(handles=handles, loc="outside upper center", ncol=2, fontsize=5)
    pubfig.save(fig, FIGDIR / "276m_all_alphabets_summary")
    return {
        "n_alphabets": table.height,
        "n_with_exact_loop_kmer": int((table["loop_kmers_in_cd47"] > 0).sum()),
        "n_seed_extend_reaching": int(
            (table["seed_extend_settings_reaching_loop"] > 0).sum()
        ),
    }


def main() -> None:
    pubfig.use_style()
    FIGDIR.mkdir(parents=True, exist_ok=True)
    print("276l", figure_prespecified())
    print("276m", figure_all_alphabets())
    print(f"\nFigures in {FIGDIR}")


if __name__ == "__main__":
    main()
