#!/usr/bin/env python3
"""Figure 276t: CD47's rank for a P66 query across every setting, and the control.

  a  CD47's rank among the 19_705 canonical human proteins, one point per
     setting, against how many proteins share rank 1 under that setting. A rank
     near 1 means nothing when thousands tie there, so both axes are needed to
     read a point.
  b  the control. The same sweep run for 200 arbitrary human proteins, each
     taking its best rank over all settings. CD47's best rank is marked. If
     arbitrary proteins reach rank 1 somewhere too, CD47 reaching it somewhere
     says nothing about CD47.

Run with the project interpreter:
  /Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3 \
      scripts/nb276_rank_sweep_figure.py
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
C_SEGMENT = pf.OKABE_ITO["blue"]
C_LOOP = pf.OKABE_ITO["bluish_green"]


def main() -> None:
    import matplotlib.pyplot as plt

    pf.use_style()
    FIGDIR.mkdir(parents=True, exist_ok=True)
    settings = pl.read_csv(OUTDIR / "34_cd47_rank_across_settings.csv")
    control = pl.read_csv(OUTDIR / "35_pretend_target_best_ranks.csv")
    # A longer protein has more offsets to find its best window in, so rank is
    # not length-neutral. The control is reported whole and again restricted to
    # proteins within 10% of CD47's length, which is the stricter comparison.
    from nb276_hsdm17_prespecified_test import HUMAN_PROTEOME, read_fasta

    human = read_fasta(HUMAN_PROTEOME)
    cd47_length = len(human["CD47"])
    control = control.with_columns(
        pl.col("protein").replace_strict(
            {n: len(s) for n, s in human.items()}, default=None
        ).alias("length")
    ).with_columns(
        (
            (pl.col("length") >= int(cd47_length * 0.9))
            & (pl.col("length") <= int(cd47_length * 1.1))
        ).alias("length_matched")
    )
    control.write_csv(OUTDIR / "35_pretend_target_best_ranks.csv")
    n_proteins = len(human)

    fig, axes = pf.figure(pf.TWO_COLUMN_MM, 66, ncols=2)

    ax = axes[0]
    for query, colour in (
        (settings["query"].unique().sort()[0], C_LOOP),
        (settings["query"].unique().sort()[1], C_SEGMENT),
    ):
        sub = settings.filter(pl.col("query") == query)
        for metric, marker in (
            ("best ungapped match count", "o"),
            (sub["metric"].unique().sort()[-1], "^"),
        ):
            rows = sub.filter(pl.col("metric") == metric)
            if rows.is_empty():
                continue
            ax.scatter(
                rows["cd47_rank"], np.maximum(rows["n_at_rank_1"], 1), s=14,
                marker=marker, color=colour, edgecolor="black", zorder=3,
                label=f"{query.split()[0]}, "
                      f"{'match count' if marker == 'o' else 'summed IDF'}",
            )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(f"CD47's rank among {n_proteins:_} human proteins")
    ax.set_ylabel("Proteins sharing rank 1 under that setting")
    ax.axvline(1, color="black", lw=0.5, ls=":")
    pf.panel_label(ax, "a")
    ax.set_title("A point at the left edge is only meaningful if it is\n"
                 "also at the bottom: rank 1 shared with nobody.",
                 fontsize=5, pad=2)

    ax = axes[1]
    best = int(settings["cd47_rank"].min())
    ranks = control["best_rank_over_all_settings"].to_numpy()
    bins = np.logspace(0, np.log10(max(ranks.max(), best) + 1), 24)
    ax.hist(ranks, bins=bins, color=C_BG, edgecolor="white", linewidth=0.3,
            label=f"{control.height} arbitrary human proteins")
    ax.axvline(best, color=C_CD47, lw=1.2, label=f"CD47 (best rank {best})")
    ax.set_xscale("log")
    ax.set_xlabel("Best rank that protein reaches in any setting")
    ax.set_ylabel("Proteins")
    pf.panel_label(ax, "b")
    reach_1 = int((ranks == 1).sum())
    matched = control.filter(pl.col("length_matched"))
    matched_reach_1 = int(
        (matched["best_rank_over_all_settings"] == 1).sum()
    )
    ax.set_title(
        f"{reach_1} of {control.height} arbitrary proteins reach rank 1 in at\n"
        f"least one setting ({matched_reach_1} of {matched.height} within 10% of\n"
        f"CD47's length); median best rank {int(np.median(ranks))}.",
        fontsize=5, pad=2,
    )

    pf.shared_legend(fig, ncol=3)
    pf.save(fig, FIGDIR / "276t_cd47_rank_across_settings")
    print("276t", {
        "n_settings": settings.height,
        "cd47_best_rank": best,
        "cd47_settings_at_rank_1": int((settings["cd47_rank"] == 1).sum()),
        "min_n_at_rank_1_where_cd47_is_first": int(
            settings.filter(pl.col("cd47_rank") == 1)["n_at_rank_1"].min()
        ) if (settings["cd47_rank"] == 1).any() else None,
        "control_reaching_rank_1": reach_1,
        "control_n": control.height,
        "control_median_best_rank": int(np.median(ranks)),
        "control_length_matched_n": matched.height,
        "control_length_matched_reaching_rank_1": matched_reach_1,
        "control_length_matched_median_best_rank": int(
            matched["best_rank_over_all_settings"].median()
        ) if matched.height else None,
    })


if __name__ == "__main__":
    main()
