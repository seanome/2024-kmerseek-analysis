#!/usr/bin/env python3
"""Figure: where BCL2 (for Ced9) and CD47 (for P66) rank among human proteins, before and
after seanome/kmerseek PR 136 scales the whole-protein chance count with target size.

Reads notebooks/241_results/length_scaled_ranks_10_settings.csv (written by
241_length_scaled_check.py), prints the table it draws, and writes
figures/241_length_scaled_ranks_10_settings.{pdf,svg,png}.

One row per alphabet and k, grouped by alphabet size. Each measure is a marker at the
middle of the partner's tie, with a bar over the tied ranks when there are ties. Left is
better: rank 1 is the top hit.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
from matplotlib.lines import Line2D

import pubfig as pf

HERE = Path(__file__).resolve().parent
DATA = HERE / "241_results" / "length_scaled_ranks_10_settings.csv"
OUT = HERE.parent / "figures" / "241_length_scaled_ranks_10_settings"

# Rows top to bottom, grouped by alphabet size (2-3 letters, 4-8, 12-18) with a gap.
GROUPS = [
    [
        ("hp_lehninger2", 17),
        ("hp_lehninger_hpc3", 14),
        ("hp_lehninger_hpc3", 18),
        ("hp_lehninger_hpc3", 21),
    ],
    [("polarity4", 9), ("gbmr4", 16), ("wwmj5", 9), ("dayhoff6", 15)],
    [("sdm12", 6), ("wass14", 6)],
]
C = pf.OKABE_ITO
# (build, metric) -> legend label, colour, marker, vertical offset inside the row
MEASURES = [
    (
        ("main", "protein Poisson p-value"),
        "whole-protein p-value, kmerseek main",
        C["vermillion"],
        "s",
        -0.24,
    ),
    (
        ("PR 136", "protein Poisson p-value"),
        "whole-protein p-value, size-scaled (PR 136)",
        C["blue"],
        "o",
        -0.12,
    ),
    (
        ("PR 136", "protein enrichment"),
        "whole-protein enrichment, size-scaled (PR 136)",
        C["orange"],
        "v",
        0.0,
    ),
    (("PR 136", "E-value"), "region E-value", C["bluish_green"], "^", 0.12),
    (("PR 136", "mean IDF"), "region mean IDF", C["reddish_purple"], "D", 0.24),
]
PARTNER = {"Ced9": "BCL2", "P66": "CD47"}


def row_positions() -> dict[tuple[str, int], float]:
    pos, y = {}, 0.0
    for group in GROUPS:
        for setting in group:
            pos[setting] = y
            y += 1
        y += 0.6  # gap between alphabet-size groups
    return pos


def main() -> None:
    pl.Config.set_tbl_rows(120)
    pl.Config.set_tbl_width_chars(160)
    pl.Config.set_fmt_str_lengths(50)
    r = pl.read_csv(DATA, infer_schema_length=None)
    pos = row_positions()
    x_missing = 40_000  # the column for "no value", right of every rank

    pf.use_style()
    fig, axs = pf.figure(pf.TWO_COLUMN_MM, 95, ncols=2, sharey=True)
    for ax, (query, partner) in zip(axs, PARTNER.items()):
        sub = r.filter(pl.col("query") == query)
        hit = set(
            sub.filter(pl.col("partner_found")).select("alphabet", "ksize").iter_rows()
        )
        rows = []
        n_hit = dict(
            ((a, k), n)
            for a, k, n in sub.select("alphabet", "ksize", "n_targets")
            .unique()
            .iter_rows()
        )
        for setting, y in pos.items():
            if setting not in hit:
                n = n_hit.get(setting, 0)
                ax.text(
                    3,
                    y,
                    (
                        f"the one protein hit is not {partner}"
                        if n == 1
                        else f"{partner} not among the {n:,} proteins hit"
                    ),
                    va="center",
                )
        for (build, metric), label, colour, marker, dy in MEASURES:
            m = sub.filter((pl.col("build") == build) & (pl.col("metric") == metric))
            for a, k, rank, tied, n in m.select(
                "alphabet", "ksize", "rank", "n_tied", "n_targets"
            ).iter_rows():
                y = pos[(a, k)] + dy
                if (a, k) not in hit:
                    continue
                if rank is None:
                    ax.plot(x_missing, y, ".", color=pf.GREY)
                    rows.append((a, k, label, None, None, n))
                    continue
                mid = rank + tied / 2
                if tied:
                    ax.plot([rank, rank + tied], [y, y], "-", color=colour, lw=0.8)
                ax.plot(mid, y, marker, color=colour, ms=3.5, label=label)
                rows.append((a, k, label, rank, tied, n))
        # Number of proteins hit: the partner's last possible place.
        for setting, y in pos.items():
            if setting in hit:
                ax.plot(n_hit[setting], y, "|", color="black", ms=6)
        ax.set_xscale("log")
        ax.set_xlim(0.7, 70_000)
        ax.set_xticks([1, 10, 100, 1_000, 10_000, x_missing])
        ax.set_xticklabels(["1", "10", "100", "1,000", "10,000", "no\nvalue"])
        ax.minorticks_off()
        ax.set_xlabel(
            f"Rank of {partner} among the human proteins {query} hits (1 = top hit)"
        )
        ax.set_title(f"query {query}, target GENCODE v49 human proteome", loc="left")
        for y in pos.values():
            ax.axhline(y, color="#DDDDDD", lw=0.3, zorder=0)
        print(f"\n{query} -> {partner}: rank (tied), proteins hit")
        print(
            pl.DataFrame(
                rows,
                schema=["alphabet", "k", "measure", "rank", "n_tied", "n_hit"],
                orient="row",
            ).sort("alphabet", "k")
        )

    axs[0].set_yticks(list(pos.values()))
    axs[0].set_yticklabels([f"{a} k={k}" for a, k in pos])
    axs[0].invert_yaxis()
    axs[0].set_ylabel("kmerseek alphabet and k-mer size")
    handles = [
        Line2D([], [], ls="", marker=mk, color=col, ms=3.5, label=lab)
        for _, lab, col, mk, _ in MEASURES
    ] + [
        Line2D(
            [], [], ls="-", color="black", lw=0.8, label="ranks tied with the partner"
        ),
        Line2D(
            [],
            [],
            ls="",
            marker="|",
            color="black",
            ms=6,
            label="number of human proteins hit (last place)",
        ),
        Line2D(
            [],
            [],
            ls="",
            marker=".",
            color=pf.GREY,
            label="partner hit, but this measure has no value (no E-value)",
        ),
    ]
    fig.legend(handles=handles, loc="outside upper center", ncol=3)
    for ax, letter in zip(axs, "ab"):
        pf.panel_label(ax, letter)
    pf.save(fig, OUT)
    print(f"\nwrote {OUT}.pdf/.svg/.png")


if __name__ == "__main__":
    main()
