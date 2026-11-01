"""Figures for the three-case alphabet sweep (notebook 241).

Data comes from 241_alphabet_ranking_collect.py: arms.csv and ranks.csv under
/Users/olga/data/botryllus/alphabet-ranking-three-cases/.

A setting is one alphabet at one k-mer size k. Most figures are a dot matrix: one row
per alphabet, seed information in bits along x, one dot per setting. Rows are grouped by how many letters the alphabet has, with a blank
row between groups, and a faint guide line runs through each row at the same height as
that alphabet's tick, so a dot can be traced back to its name.

Marks mean the same thing in every figure here:

* a filled dot, coloured on one purple ramp, is the quantity that figure is about;
* a grey cross means there is nothing at that setting (no region, or the setting was not run);
* a small black dot means the setting has no E-value, because the Karlin-Altschul fit was
  refused;
* a black ring around a filled dot marks the one setting feature named in that figure's
  legend.

The legend sits above the panels, in a band only as tall as the legend itself.
"""

from __future__ import annotations

import io
import os
import urllib.request
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.lines import Line2D

import pubfig as pf
from hp_conservation_utils import ALPHABET_CLUSTERS, finish_figure

# Every figure here is drawn at its printed size (183 mm, two columns) with the Nature
# style: Arial, 6-7 pt text, 0.5 pt axes. Sizes are not set by hand except the 5 pt
# numbers written inside dots.
pf.use_style()
DOT_PT = 5
MARK_S = 16  # scatter area of one setting's dot, pt^2

# NB241_DIR is set by 241_alphabet_ranking.sbatch on Sherlock; the default is the laptop
# folder that holds the small tables copied back for this notebook.
DATA = Path(
    os.environ.get(
        "NB241_DIR", "/Users/olga/data/botryllus/alphabet-ranking-three-cases"
    )
)
FIG = Path(__file__).resolve().parent.parent / "figures"

N_HUMAN = 19_732
TOOLS = (
    "kmerseek 0.4.0 (olgabot/ka-lambda-per-region 982a055) index + search + pair; "
    "target GENCODE v49 canonical human proteins (19_732), scaled 1, low-complexity "
    "k-mers removed; region extension at each alphabet's kappa-optimal mismatch "
    "penalty, Karlin-Altschul fit per index"
)

CLASSES = {
    "hp_lehninger2": 2,
    "hp_thomas_dill2": 2,
    "hp_kyte_doolittle2": 2,
    "hp_thomas_dill_no_c2": 2,
    "hp_lehninger_c_nonpolar2": 2,
    "hp_pbotc_1st_ed2": 2,
    "hp_lehninger_hpc3": 3,
    "gbmr4": 4,
    "polarity4": 4,
    "wwmj5": 5,
    "dayhoff6": 6,
    "gbmr7": 7,
    "funcgroups8": 8,
    "sdm12": 12,
    "mmseqs12": 12,
    "wass14": 14,
    "hsdm17": 17,
    "uniprot18": 18,
    "protein20": 20,
}
ORDER = sorted(CLASSES, key=lambda a: (CLASSES[a], a))

# Rows are grouped by how many letters the alphabet has, with one blank row between
# groups, so a group can be read as a block without counting rows.
GROUPS = [
    ("2 to 3 letters", lambda n: n <= 3),
    ("4 to 8 letters", lambda n: 4 <= n <= 8),
    ("12 to 18 letters", lambda n: 12 <= n <= 18),
    ("all 20 letters", lambda n: n == 20),
]
GROUP_GAP = 1.0


def _row_positions() -> tuple[dict[str, float], float, list[tuple[str, float, float]]]:
    rows: dict[str, float] = {}
    spans: list[tuple[str, float, float]] = []
    y = 0.0
    for i, (name, keep) in enumerate(GROUPS):
        if i:
            y += GROUP_GAP
        members = [a for a in ORDER if keep(CLASSES[a])]
        top = y
        for a in members:
            rows[a] = y
            y += 1.0
        spans.append((name, top, y - 1.0))
    return rows, y - 1.0, spans


ROW, Y_LAST, GROUP_SPANS = _row_positions()

METRICS5 = ["E-value", "mean IDF", "tf-idf", "enrichment", "Poisson p-value"]

# One hue for magnitude. Truncated so the light end is still visible on white.
# RAMP: more = darker. RANK_RAMP: rank 1 = darkest, so the best result is the mark the
# eye lands on.
RAMP = mpl.colors.LinearSegmentedColormap.from_list(
    "count", [plt.get_cmap("Purples")(x) for x in np.linspace(0.35, 1.0, 256)]
)
RANK_RAMP = RAMP.reversed()
NOT_FOUND = "#9a9a9a"
# "No E-value here" is a small black dot, not an open circle: an open circle read as one
# more of the filled circles around it.
NO_EVALUE = "#101010"
NO_EVALUE_MARKER = dict(marker=".", s=6, color=NO_EVALUE, lw=0)
GUIDE = "#ececec"


def _no_evalue_handle(label: str) -> Line2D:
    return Line2D([], [], marker=".", ls="", ms=4.5, color=NO_EVALUE, label=label)


# The existing analysis excludes both lambda-dependent metrics when database-level
# drift is nonnegative or within 0.001 of zero. This is a heuristic exclusion:
# gbmr4 has slightly negative drift, which can have a positive mathematical root.
SCORE_METRICS = ("E-value", "bit score")
DRIFT_TOLERANCE = 0.001

# Computed once per protein pair, by kmerseek 982a055, which is from before
# seanome/kmerseek PR 136 (merged 2026-10-06) scaled the chance count of shared k-mers
# with the target's length. These ranks change after PR 136.
WHOLE_PROTEIN_METRICS = ("containment", "protein enrichment", "protein Poisson p-value")


def no_evalue_alphabets(arms: pl.DataFrame) -> list[str]:
    """Alphabets excluded by the existing nonnegative-or-near-zero drift guard.

    This selection rule does not prove that a positive lambda is impossible.
    """
    return sorted(
        arms.group_by("alphabet")
        .agg(pl.col("chance_drift").first())
        .filter(pl.col("chance_drift") >= -DRIFT_TOLERANCE)["alphabet"]
    )


def load() -> tuple[pl.DataFrame, pl.DataFrame]:
    """arms.csv and ranks.csv, with the E-value and bit score emptied for the alphabets
    excluded by the drift guard (`valid` = False on those rows).

    A rank is over the targets with a finite value under that metric (`n_finite`), so a
    target with no E-value is left out, not ranked last. `n_targets` is every target
    with a region. `percent` is 100 * rank / n_finite.
    """
    # The first hundred rows of ranks.csv can be all-null in `rank`, so the types are
    # given rather than inferred.
    arms = pl.read_csv(DATA / "arms.csv", infer_schema_length=None)
    ranks = pl.read_csv(
        DATA / "ranks.csv",
        schema_overrides={
            "rank": pl.Int64,
            "n_tied": pl.Int64,
            "partner_value": pl.Float64,
            "best_value": pl.Float64,
            "partner_found": pl.Boolean,
            "top_gene": pl.Utf8,
            "n_top_tied": pl.Int64,
            "n_finite": pl.Int64,
        },
    )
    bad = no_evalue_alphabets(arms)
    valid = ~(pl.col("metric").is_in(SCORE_METRICS) & pl.col("alphabet").is_in(bad))
    ranks = ranks.with_columns(valid.alias("valid")).with_columns(
        [
            pl.when(pl.col("valid")).then(pl.col(c)).alias(c)
            for c in (
                "rank",
                "n_tied",
                "partner_value",
                "best_value",
                "top_gene",
                "n_top_tied",
            )
        ]
        + [
            pl.when(pl.col("valid"))
            .then(pl.col("n_finite"))
            .otherwise(0)
            .alias("n_finite"),
            (pl.col("partner_found") & pl.col("valid")).alias("partner_found"),
        ]
    )
    ranks = ranks.with_columns(
        (100 * pl.col("rank") / pl.col("n_finite")).alias("percent")
    )
    return arms, ranks


def _figure_with_legend_row(
    ncols: int,
    size_mm: tuple,
    handles: list,
    sharey: bool = True,
    width_ratios=None,
    legend_fontsize: float = 6.0,
    title_in: float = 0.16,
    xlabel_in: bool = True,
    wspace: float | None = None,
    nrows: int = 1,
):
    """A figure whose top row is an axes holding only the legend, so the legend reads
    before the marks. The band is only as tall as the legend text, so there is no gap
    between the TOOLS line above the figure and the legend."""
    w_in, h_in = size_mm[0] * pf.MM, size_mm[1] * pf.MM
    fig = plt.figure(figsize=(w_in, h_in))
    n_lines = len(handles)
    # The legend band is only as tall as the legend text, and the panels start right
    # under it. Margins are set here rather than left to tight_layout, which refuses to
    # run on this layout and would leave matplotlib's default 0.88 top margin: a band of
    # white between the TOOLS line above the figure and the legend.
    band_in = n_lines * legend_fontsize * 1.6 / 72.0 + 0.03
    top, bottom = 0.995, min(0.25, (0.32 + 0.12 * bool(xlabel_in)) / h_in)
    left, right = min(0.30, 1.30 / w_in), 0.99
    gap = title_in / h_in
    span = top - bottom
    gs = fig.add_gridspec(
        1 + nrows,
        ncols,
        height_ratios=[band_in] + [max(h_in - band_in, 1.0) / nrows] * nrows,
        width_ratios=width_ratios,
        top=top,
        bottom=bottom,
        left=left,
        right=right,
        hspace=2 * gap / max(span - gap, 0.05),
        wspace=wspace,
    )
    lax = fig.add_subplot(gs[0, :])
    lax.axis("off")
    lax.set_navigate(False)
    lax.legend(
        handles=handles,
        loc="upper left",
        fontsize=legend_fontsize,
        frameon=False,
        borderaxespad=0.0,
        handletextpad=0.4,
        labelspacing=0.35,
        borderpad=0.0,
    )
    axes = []
    for i in range(ncols * nrows):
        ax = fig.add_subplot(
            gs[1 + i // ncols, i % ncols], sharey=axes[0] if (axes and sharey) else None
        )
        if axes and sharey:
            plt.setp(ax.get_yticklabels(), visible=False)
        axes.append(ax)
    return fig, axes


def _finish(fig, path: Path, **kw) -> None:
    """finish_figure at this module's printed size: header and footer at 6-7 pt,
    450 dpi, lines wrapped to the 183 mm width."""
    finish_figure(fig, path, font_scale=0.72, dpi=450, wrap=165, **kw)


def _row_ticks(ax, labels: bool = True):
    """The y axis of an alphabet-row panel: one tick per alphabet, blank rows between
    the letter-count groups, and a guide line at the height of each tick."""
    ax.set_yticks([ROW[a] for a in ORDER])
    if labels:
        ax.set_yticklabels([f"{a} ({CLASSES[a]})" for a in ORDER])
    ax.set_ylim(Y_LAST + 0.9, -0.9)
    for a in ORDER:
        ax.axhline(ROW[a], color=GUIDE, lw=0.8, zorder=0)


def _rows_axis(ax, xlabel: bool = True):
    _row_ticks(ax)
    ax.set_xlim(13, 47)
    ax.set_xticks([16, 20, 24, 28, 32, 36, 40, 44])
    if xlabel:
        ax.set_xlabel("information in one seed (bits = k × bits per position)")
    ax.grid(axis="x", alpha=0.25, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)


def fit_status_figure(
    arms: pl.DataFrame, path: Path, hypothesis: str, conclusion: str
) -> pl.DataFrame:
    """One panel: which arms got a Karlin-Altschul fit, and so have an E-value at all."""
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            ls="",
            ms=4,
            color=RAMP(0.85),
            label="the Karlin-Altschul fit worked: this setting has E-values",
        ),
        _no_evalue_handle(
            "the fit was refused even after the retry: this setting has every metric except the E-value"
        ),
    ]
    if arms.filter(pl.col("fitted").is_null()).height:
        handles.append(
            Line2D(
                [],
                [],
                marker="x",
                ls="",
                ms=3.5,
                color=NOT_FOUND,
                label="the setting was not run",
            )
        )
    fig, axes = _figure_with_legend_row(1, (183, 85), handles)
    ax = axes[0]
    for row in arms.iter_rows(named=True):
        y, x = ROW[row["alphabet"]], row["bits"]
        if row["fitted"] is True:
            ax.scatter(
                x, y, s=MARK_S, color=RAMP(0.85), edgecolors="white", lw=0.4, zorder=4
            )
        elif row["fitted"] is False:
            ax.scatter(x, y, zorder=4, **NO_EVALUE_MARKER)
        else:
            ax.scatter(x, y, s=8, marker="x", color=NOT_FOUND, lw=1.0, zorder=3)
    ax.set_title("which settings have an E-value")
    _rows_axis(ax)
    _finish(
        fig,
        path,
        tight_layout=False,
        tools=TOOLS,
        hypothesis=hypothesis,
        conclusion=conclusion,
        header_y=1.01,
        footer_y=-0.02,
    )
    return (
        arms.with_columns(
            pl.when(pl.col("fitted") == True)
            .then(pl.lit("has an E-value"))  # noqa: E712
            .when(pl.col("fitted") == False)
            .then(pl.lit("no E-value: fit refused"))  # noqa: E712
            .otherwise(pl.lit("not run"))
            .alias("fit")
        )
        .group_by("alphabet", "fit")
        .agg(
            pl.len().alias("settings"),
            pl.col("bits").min().round(0).alias("from_bits"),
            pl.col("bits").max().round(0).alias("to_bits"),
        )
        .sort("alphabet", "fit")
    )


def rank_matrix(
    ranks: pl.DataFrame,
    arms: pl.DataFrame,
    query: str,
    partner: str,
    path: Path,
    title_first_line: str,
    hypothesis: str,
    conclusion: str,
    annotate_le: int = 10,
) -> pl.DataFrame:
    """Five panels, one per metric: where `partner` ranks among human targets when
    `query` is searched, per alphabet (rows) and seed information (x)."""
    sub = ranks.filter((pl.col("query") == query) & pl.col("metric").is_in(METRICS5))
    fit = arms.select("alphabet", "ksize", "fitted", "searched")
    sub = sub.join(fit, on=["alphabet", "ksize"], how="left")
    norm = mpl.colors.LogNorm(vmin=1, vmax=10_000)

    handles = [
        Line2D(
            [],
            [],
            marker="o",
            ls="",
            ms=4,
            color=RANK_RAMP(0.15),
            label=f"{partner} is among the hits; colour = its rank, number written when rank ≤ {annotate_le}",
        ),
        Line2D(
            [],
            [],
            marker="x",
            ls="",
            ms=3.5,
            color=NOT_FOUND,
            label=f"{partner} is not among the hits at this setting",
        ),
        _no_evalue_handle(
            "E-value panel only: no E-value at this setting (the Karlin-Altschul fit was refused, "
            "or excluded by the drift guard: gbmr4, gbmr7, hp_lehninger_hpc3)"
        ),
    ]
    fig, axes = _figure_with_legend_row(5, (183, 95), handles)
    for ax, metric in zip(axes, METRICS5):
        m = sub.filter(pl.col("metric") == metric)
        for row in m.iter_rows(named=True):
            y = ROW[row["alphabet"]]
            x = row["bits"]
            if not row["searched"]:
                continue
            if metric == "E-value" and (row["fitted"] is not True or not row["valid"]):
                ax.scatter(x, y, zorder=3, **NO_EVALUE_MARKER)
                continue
            if not row["partner_found"]:
                ax.scatter(x, y, s=8, marker="x", color=NOT_FOUND, lw=1.0, zorder=3)
                continue
            r = row["rank"]
            ax.scatter(
                x,
                y,
                s=MARK_S,
                c=[RANK_RAMP(norm(r))],
                edgecolors="white",
                lw=0.4,
                zorder=4,
            )
            if r <= annotate_le:
                ax.text(
                    x,
                    y,
                    str(r),
                    ha="center",
                    va="center",
                    fontsize=DOT_PT,
                    color="white",
                    zorder=5,
                    fontweight="bold",
                )
        ax.set_title(metric)
        _rows_axis(ax, xlabel=(metric == METRICS5[2]))

    cb = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=RANK_RAMP),
        ax=axes,
        fraction=0.012,
        pad=0.01,
        aspect=30,
    )
    cb.set_label(f"rank of {partner} among the human proteins ranked (1 = best)")
    cb.set_ticks([1, 10, 100, 1000, 10_000])
    cb.set_ticklabels(["1", "10", "100", "1_000", "10_000"])

    _finish(
        fig,
        path,
        tight_layout=False,
        tools=TOOLS,
        hypothesis=hypothesis,
        conclusion=conclusion,
        title=title_first_line,
        header_y=1.01,
        footer_y=-0.02,
    )
    return sub


def best_per_alphabet(ranks: pl.DataFrame, query: str) -> pl.DataFrame:
    """For each alphabet, the best rank the partner reaches over every k and every one
    of the five metrics, and which setting and metric got there."""
    sub = ranks.filter(
        (pl.col("query") == query)
        & pl.col("metric").is_in(METRICS5)
        & pl.col("partner_found")
        & pl.col("rank").is_not_null()
    ).sort("rank", "n_tied", "bits")
    best = (
        sub.group_by("alphabet")
        .first()
        .select(
            "alphabet",
            "rank",
            "n_tied",
            "n_finite",
            "n_targets",
            "ksize",
            "bits",
            "metric",
            "partner_value",
        )
    )
    found_any = sub.group_by("alphabet").agg(
        pl.col("bits").min().alias("found_from_bits"),
        pl.col("bits").max().alias("found_to_bits"),
    )
    out = pl.DataFrame({"alphabet": ORDER, "classes": [CLASSES[a] for a in ORDER]})
    return out.join(best, on="alphabet", how="left").join(
        found_any, on="alphabet", how="left"
    )


def best_rank_figure(
    ranks: pl.DataFrame, path: Path, hypothesis: str, conclusion: str
) -> pl.DataFrame:
    """Two panels: the best rank BCL2 (gold standard) and CD47 (P66/CD47, a proposed pair) ever reach,
    per alphabet, over every k and metric."""
    handles = [
        Line2D(
            [],
            [],
            marker="s",
            ls="",
            ms=4.5,
            color=RANK_RAMP(0.5),
            label="bar: best rank reached, over every k and every one of the five metrics; label says which",
        ),
        Line2D(
            [],
            [],
            marker="x",
            ls="",
            ms=3.5,
            color=NOT_FOUND,
            label="the partner is never among the hits for this alphabet at any k",
        ),
    ]
    fig, axes = _figure_with_legend_row(2, (183, 85), handles, wspace=0.55)
    tables = []
    for ax, (q, partner, label) in zip(
        axes,
        [
            ("Ced9", "BCL2", "gold standard: Ced9 query, BCL2 is the known homolog"),
            (
                "P66",
                "CD47",
                "P66/CD47, a proposed pair: P66 query, CD47 is the proposed partner",
            ),
        ],
    ):
        b = best_per_alphabet(ranks, q)
        tables.append(b.with_columns(pl.lit(q).alias("query")))
        for row in b.iter_rows(named=True):
            y = ROW[row["alphabet"]]
            if row["rank"] is None:
                ax.scatter(1.0, y, marker="x", color=NOT_FOUND, s=9, zorder=3)
                ax.text(
                    1.25,
                    y,
                    "never among the hits",
                    va="center",
                    color=NOT_FOUND,
                )
                continue
            r = row["rank"]
            ax.barh(y, r, left=1, height=0.55, color=RANK_RAMP(0.5), zorder=2)
            ax.text(
                r * 1.12,
                y,
                f"{r:_} of {row['n_finite']:_}  ({row['metric']}, k={row['ksize']}"
                + (f", {row['n_tied']} tied)" if row["n_tied"] else ")"),
                va="center",
            )
        ax.set_xscale("log")
        ax.set_xlim(1, 1e8)
        ax.set_xticks([1, 10, 100, 1000, 10_000])
        ax.set_xticklabels(["1", "10", "100", "1_000", "10_000"])
        ax.set_xlabel(
            f"best rank of {partner} among the human proteins ranked (1 = best)",
        )
        ax.set_title(label)
        _row_ticks(ax)
        ax.grid(axis="x", alpha=0.25, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
    _finish(
        fig,
        path,
        tight_layout=False,
        tools=TOOLS,
        hypothesis=hypothesis,
        conclusion=conclusion,
        header_y=1.01,
        footer_y=-0.02,
    )
    return pl.concat(tables)


def null_rank_figure(
    nulls: pl.DataFrame, path: Path, hypothesis: str, conclusion: str
) -> pl.DataFrame:
    """One panel: the best rank BCL2 and CD47 reach anywhere in the sweep, against two
    nulls for the best rank of a human protein over the same settings and metrics
    (best_of_many_nulls); the number right of each bar is the share of null draws as
    good as or better than the partner."""
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            ls="",
            ms=4,
            color=RANK_RAMP(0.25),
            label="best rank of the partner, over 19 alphabets, every k and the five metrics",
        ),
        Line2D(
            [],
            [],
            marker="s",
            ls="",
            ms=4.5,
            color="#d0d0d0",
            label="middle 90% of the best rank of a random human protein, 20_000 draws",
        ),
        Line2D([], [], marker="s", ls="", ms=4.5, color="#f0dcae", label="the top 10"),
        Line2D(
            [],
            [],
            ls="",
            label=f"right column: % of random draws as good or better; "
            f"beating chance needs {BEATS_CHANCE_PERCENT}% or less",
        ),
    ]
    fig, (ax,) = _figure_with_legend_row(1, (183, 62), handles)
    ax.axvspan(0.7, 10, color="#f0dcae", zorder=1)
    _null_panel(ax, list(nulls.iter_rows(named=True)), labels=True, text_x=1e5)
    ax.set_xlim(0.7, 1e5)
    ax.set_xticks([1, 10, 100, 1000, 10_000])
    ax.set_xticklabels(["1", "10", "100", "1_000", "10_000"])
    ax.set_xlabel(
        "best rank among the human proteins with a value (1 = top of the list)"
    )
    _finish(
        fig,
        path,
        tight_layout=False,
        tools=TOOLS,
        hypothesis=hypothesis,
        conclusion=conclusion,
        header_y=1.01,
        footer_y=-0.04,
    )
    return nulls


def best_per_metric(ranks: pl.DataFrame, query: str) -> pl.DataFrame:
    """For every metric the search writes, the best the partner ever does, as a percent
    of the proteins with a value at that setting. A rank of 213 out of 18_064 and one of 173 out of
    1_161 are not the same result, so the comparison is on the percent, not the rank."""
    return (
        ranks.filter(
            (pl.col("query") == query)
            & pl.col("partner_found")
            & pl.col("rank").is_not_null()
        )
        .with_columns(pl.col("percent").alias("best_percent"))
        .sort("best_percent", "rank")
        .group_by("metric")
        .first()
        .select(
            "metric",
            "best_percent",
            "rank",
            "n_finite",
            "n_targets",
            "n_tied",
            "alphabet",
            "ksize",
        )
        .sort("best_percent", "metric")
    )


def metric_sweep_figure(
    ranks: pl.DataFrame, path: Path, hypothesis: str, conclusion: str
) -> pl.DataFrame:
    """Two panels: every metric the search writes, and the best the known partner ever
    does under it, anywhere in the sweep."""
    top10_pct = 100 * 10 / N_HUMAN
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            ls="",
            ms=4.5,
            color=RANK_RAMP(0.25),
            label="best the partner ever does under this metric, as a percent of the proteins with a value at that setting",
        ),
        Line2D(
            [],
            [],
            marker="o",
            ls="",
            ms=4.5,
            markerfacecolor=RANK_RAMP(0.25),
            markeredgecolor="black",
            markeredgewidth=0.8,
            label="that best rank is a tie: other proteins have exactly the same value, and the order among them is arbitrary",
        ),
        Line2D(
            [],
            [],
            marker="s",
            ls="",
            ms=4.5,
            color="#d9b25a",
            label=f"where the top 10 of the human proteome would sit ({top10_pct:.2f}%)",
        ),
    ]
    fig, axes = _figure_with_legend_row(
        1, (183, 125), handles, sharey=False, nrows=2, title_in=0.5
    )
    tables = []
    for ax, (q, partner) in zip(axes, [("Ced9", "BCL2"), ("P66", "CD47")]):
        b = best_per_metric(ranks, q)
        tables.append(b.with_columns(pl.lit(q).alias("query")))
        ax.axvline(top10_pct, color="#d9b25a", lw=1.5, zorder=1)
        for i, row in enumerate(b.iter_rows(named=True)):
            tied = (row["n_tied"] or 0) > 0
            ax.scatter(
                row["best_percent"],
                i,
                s=24,
                color=RANK_RAMP(0.25),
                edgecolors="black" if tied else "white",
                lw=0.8 if tied else 0.8,
                zorder=4,
            )
            label = (
                f"  {row['rank']:_} of {row['n_finite']:_} ({row['alphabet']} k={row['ksize']}"
                + (f", {row['n_tied']:_} tied)" if tied else ")")
            )
            ax.text(row["best_percent"] * 1.2, i, label, va="center")
        ax.set_yticks(range(b.height))
        ax.set_yticklabels(
            [
                f"{m} (before PR 136)" if m in WHOLE_PROTEIN_METRICS else m
                for m in b["metric"]
            ]
        )
        ax.set_ylim(b.height - 0.4, -0.7)
        ax.set_xscale("log")
        ax.set_xlim(0.03, 4_000)
        ax.set_xticks([0.05, 0.1, 1, 10, 100])
        ax.set_xticklabels(["0.05", "0.1", "1", "10", "100"])
        if q == "P66":
            ax.set_xlabel(
                "best position of the partner among the proteins ranked (percent; smaller is better)"
            )
        ax.set_title(f"{q} → {partner}")
        ax.grid(axis="x", alpha=0.25, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        for y in range(b.height):
            ax.axhline(y, color=GUIDE, lw=0.8, zorder=0)
    _finish(
        fig,
        path,
        tight_layout=False,
        tools=TOOLS,
        hypothesis=hypothesis,
        conclusion=conclusion,
        header_y=1.01,
        footer_y=-0.04,
    )
    return pl.concat(tables)


def bhf_matrix(
    ranks: pl.DataFrame,
    arms: pl.DataFrame,
    path: Path,
    hypothesis: str,
    conclusion: str,
) -> pl.DataFrame:
    """Two panels for BHF, which has no known partner: the best E-value any human
    protein reaches, and how many human proteins have any region at all."""
    e = ranks.filter((pl.col("query") == "BHF") & (pl.col("metric") == "E-value")).join(
        arms.select("alphabet", "ksize", "fitted", "searched", "n_targets_BHF"),
        on=["alphabet", "ksize"],
        how="left",
    )
    norm_e = mpl.colors.LogNorm(vmin=1e-3, vmax=1e3)
    ramp_e = RANK_RAMP  # dark = small E-value = more surprising
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            ls="",
            ms=4,
            color=RAMP(0.75),
            label="filled dot: colour = the panel's quantity",
        ),
        Line2D(
            [],
            [],
            marker="o",
            ls="",
            ms=4,
            markerfacecolor=ramp_e(0.1),
            markeredgecolor="black",
            markeredgewidth=0.8,
            label="black ring, left panel only: best E-value below 1, gene and E written beside it",
        ),
        Line2D(
            [],
            [],
            marker="x",
            ls="",
            ms=3.5,
            color=NOT_FOUND,
            label="no human protein has a region at this setting",
        ),
        _no_evalue_handle(
            "left panel only: no E-value at this setting (fit refused, or excluded by the drift guard)"
        ),
    ]
    fig, axes = _figure_with_legend_row(2, (183, 95), handles)
    ax = axes[0]
    below_one: dict[float, list[str]] = {}
    for row in e.iter_rows(named=True):
        y, x = ROW[row["alphabet"]], row["bits"]
        if not row["searched"]:
            continue
        if row["fitted"] is not True or not row["valid"]:
            ax.scatter(x, y, zorder=3, **NO_EVALUE_MARKER)
            continue
        if row["best_value"] is None:
            ax.scatter(x, y, s=8, marker="x", color=NOT_FOUND, zorder=3)
            continue
        v = row["best_value"]
        ax.scatter(
            x,
            y,
            s=MARK_S,
            c=[ramp_e(norm_e(v))],
            edgecolors="black" if v < 1 else "white",
            lw=0.8 if v < 1 else 0.4,
            zorder=4,
        )
        if v < 1:
            below_one.setdefault(y, []).append(
                f"{row['top_gene']} E={v:.2g} at {x:.0f} bits"
            )
    # Names go in the empty space right of the last setting, with a leader from the row.
    for y, labels in below_one.items():
        ax.text(
            51.5,
            y,
            "; ".join(labels),
            va="center",
            ha="right",
            fontsize=DOT_PT,
            zorder=5,
            bbox=dict(
                boxstyle="round,pad=0.15",
                facecolor="white",
                edgecolor="none",
                alpha=0.85,
            ),
        )
    ax.set_title("best E-value of any human protein")
    _rows_axis(ax)
    ax.set_xlim(13, 52)
    cb = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm_e, cmap=ramp_e), ax=ax, fraction=0.04, pad=0.02
    )
    cb.set_label("best E-value (lower is more surprising)")

    ax = axes[1]
    norm_n = mpl.colors.LogNorm(vmin=1, vmax=20_000)
    for row in e.iter_rows(named=True):
        y, x = ROW[row["alphabet"]], row["bits"]
        if not row["searched"]:
            continue
        n = row["n_targets_BHF"] or 0
        if n == 0:
            ax.scatter(x, y, s=8, marker="x", color=NOT_FOUND, zorder=3)
        else:
            ax.scatter(
                x,
                y,
                s=MARK_S,
                c=[RAMP(norm_n(n))],
                edgecolors="white",
                lw=0.4,
                zorder=4,
            )
    ax.set_title("human proteins with at least one matched region")
    _rows_axis(ax)
    cb = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm_n, cmap=RAMP), ax=ax, fraction=0.04, pad=0.02
    )
    cb.set_label("number of human proteins hit (of 19_732)")

    _finish(
        fig,
        path,
        tight_layout=False,
        tools=TOOLS,
        hypothesis=hypothesis,
        conclusion=conclusion,
        header_y=1.01,
        footer_y=-0.02,
    )
    return e.select(
        "alphabet", "ksize", "bits", "fitted", "n_targets_BHF", "best_value", "top_gene"
    ).sort("alphabet", "ksize")


def pair_matrix(
    arms: pl.DataFrame, path: Path, hypothesis: str, conclusion: str
) -> pl.DataFrame:
    """No database at all: how many exact k-mers the two proteins of each pair share,
    per alphabet and seed information."""
    norm = mpl.colors.LogNorm(vmin=1, vmax=100)
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            ls="",
            ms=4,
            color=RAMP(0.75),
            label="the two proteins share at least one exact k-mer; colour = how many",
        ),
        Line2D(
            [],
            [],
            marker="o",
            ls="",
            ms=4,
            markerfacecolor=RAMP(0.75),
            markeredgecolor="black",
            markeredgewidth=0.8,
            label="Ced9 panel only: a shared region lies in the BH3-binding groove (Ced9 150-195, BCL2 125-170)",
        ),
        Line2D(
            [],
            [],
            marker="x",
            ls="",
            ms=3.5,
            color=NOT_FOUND,
            label="no shared k-mer at this setting",
        ),
    ]
    fig, axes = _figure_with_legend_row(2, (183, 95), handles)
    for ax, (q, partner) in zip(axes, [("Ced9", "BCL2"), ("P66", "CD47")]):
        col = f"pair_{q}_shared_kmers"
        for row in arms.iter_rows(named=True):
            y, x = ROW[row["alphabet"]], row["bits"]
            n = row[col]
            if n is None:
                continue
            if n == 0:
                ax.scatter(x, y, s=8, marker="x", color=NOT_FOUND, zorder=3)
                continue
            in_win = row.get("pair_Ced9_in_window") if q == "Ced9" else None
            ax.scatter(
                x,
                y,
                s=MARK_S,
                c=[RAMP(norm(n))],
                edgecolors="black" if in_win else "white",
                lw=0.8 if in_win else 0.4,
                zorder=4,
            )
            if n <= 9:
                ax.text(
                    x,
                    y,
                    str(n),
                    ha="center",
                    va="center",
                    fontsize=DOT_PT,
                    color="white" if n >= 4 else "#1a1a1a",
                    zorder=5,
                    fontweight="bold",
                )
        ax.set_title(f"{q} vs {partner}: exact k-mers the two proteins share")
        _rows_axis(ax)
    cb = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=RAMP), ax=axes, fraction=0.02, pad=0.02
    )
    cb.set_label("shared k-mers (number written when ≤ 9)")
    _finish(
        fig,
        path,
        tight_layout=False,
        tools=TOOLS.replace("index + search + pair", "pair (no database)"),
        hypothesis=hypothesis,
        conclusion=conclusion,
        header_y=1.01,
        footer_y=-0.02,
    )
    return arms.select(
        "alphabet",
        "ksize",
        "bits",
        "pair_Ced9_shared_kmers",
        "pair_Ced9_in_window",
        "pair_P66_shared_kmers",
    ).sort("alphabet", "ksize")


def partner_evalue_figure(
    ranks: pl.DataFrame,
    arms: pl.DataFrame,
    path: Path,
    hypothesis: str,
    conclusion: str,
) -> pl.DataFrame:
    """Two panels in the row layout: the E-value BCL2 (left) and CD47 (right) themselves
    get at every setting where they are among the hits and an E-value exists. The best one
    per alphabet is written at the right edge."""
    norm = mpl.colors.LogNorm(vmin=1e-1, vmax=1e6)
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            ls="",
            ms=4,
            color=RANK_RAMP(0.5),
            label="the partner is among the hits and has an E-value; colour = that E-value (dark = below 1)",
        ),
        Line2D(
            [],
            [],
            marker="x",
            ls="",
            ms=3.5,
            color=NOT_FOUND,
            label="the partner is not among the hits at this setting, or the setting has no E-value",
        ),
    ]
    fig, axes = _figure_with_legend_row(2, (183, 95), handles)
    tables = []
    for ax, (q, partner) in zip(axes, [("Ced9", "BCL2"), ("P66", "CD47")]):
        sub = (
            ranks.filter((pl.col("query") == q) & (pl.col("metric") == "E-value"))
            .join(
                arms.select("alphabet", "ksize", "fitted", "searched"),
                on=["alphabet", "ksize"],
                how="left",
            )
            .filter(pl.col("searched"))
        )
        best: dict[float, float] = {}
        for row in sub.iter_rows(named=True):
            y, x = ROW[row["alphabet"]], row["bits"]
            ok = (
                row["fitted"] is True
                and row["valid"]
                and row["partner_found"]
                and row["partner_value"] is not None
            )
            if not ok:
                ax.scatter(x, y, s=8, marker="x", color=NOT_FOUND, zorder=3)
                continue
            v = row["partner_value"]
            ax.scatter(
                x,
                y,
                s=MARK_S,
                c=[RANK_RAMP(norm(v))],
                edgecolors="white",
                lw=0.4,
                zorder=4,
            )
            best[y] = min(best.get(y, float("inf")), v)
        for y, v in best.items():
            ax.text(
                51.5,
                y,
                f"best E = {v:_.0f}" if v >= 10 else f"best E = {v:.2g}",
                va="center",
                ha="right",
            )
        ax.set_title(f"{partner}'s own E-value when {q} is the query")
        _rows_axis(ax)
        ax.set_xlim(13, 52)
        tables.append(
            sub.filter(
                pl.col("partner_found") & (pl.col("fitted") == True)
            ).select(  # noqa: E712
                "query",
                "alphabet",
                "ksize",
                "bits",
                "partner_value",
                "rank",
                "n_finite",
                "n_targets",
            )
        )
    cb = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=RANK_RAMP),
        ax=axes,
        fraction=0.02,
        pad=0.02,
    )
    cb.set_label(
        "E-value of the partner's best region (lower is more surprising; 1 = chance)",
    )
    _finish(
        fig,
        path,
        tight_layout=False,
        tools=TOOLS,
        hypothesis=hypothesis,
        conclusion=conclusion,
        header_y=1.01,
        footer_y=-0.02,
    )
    return pl.concat(tables)


def pr44_crosscheck_figure(
    mine: pl.DataFrame,
    theirs: pl.DataFrame | None,
    control: pl.DataFrame | None,
    cd47: pl.DataFrame,
    path: Path,
    hypothesis: str,
    conclusion: str,
):
    """Left panels, one per alphabet PR #44 ran: the exact k-mers P66 and CD47 share at
    each k, measured twice. Right panel: how many other human proteins CD47 beats, in
    PR #44's 300 random proteins and in this sweep's whole-proteome hit list."""
    have = set(theirs["alphabet"].unique()) if theirs is not None else set()
    ladder_alphabets = [a for a in ORDER if a in have] or [
        "hp_lehninger2",
        "polarity4",
        "funcgroups8",
        "protein20",
    ]
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            ls="-",
            ms=3.5,
            color=RAMP(0.85),
            label="this sweep (notebook 241), kmerseek pair on the same two proteins",
        ),
        Line2D(
            [],
            [],
            marker="s",
            ls="",
            ms=4.5,
            markerfacecolor="none",
            markeredgecolor="#c0392b",
            markeredgewidth=0.8,
            label="PR #44's ladder, the same measurement run independently; it ran k values this sweep does not",
        ),
        Line2D(
            [],
            [],
            marker="x",
            ls="",
            ms=3.5,
            color=NOT_FOUND,
            label="this sweep ran this k and the two proteins share no k-mer",
        ),
        Line2D(
            [],
            [],
            marker="s",
            ls="",
            ms=4.5,
            color="#d0d0d0",
            label="right panel: percent of PR #44's 300 random human proteins that CD47 beats on shared k-mers",
        ),
        Line2D(
            [],
            [],
            marker="o",
            ls="",
            ms=4,
            color=RANK_RAMP(0.3),
            label="right panel: percent of this sweep's 19_732-protein hit list that CD47 beats, range over the five metrics",
        ),
    ]
    n = len(ladder_alphabets)
    fig, axes = _figure_with_legend_row(
        n + 1,
        (183, 70),
        handles,
        sharey=False,
        width_ratios=[1.0] * n + [1.7],
        wspace=0.5,
    )
    ymax = max(
        1,
        int(mine["shared_kmers_241"].max() or 1),
        int(theirs["shared_kmers"].max() or 1) if theirs is not None else 1,
    )
    for i, (ax, alpha) in enumerate(zip(axes, ladder_alphabets)):
        m = mine.filter(pl.col("alphabet") == alpha).sort("k")
        pos = m.filter(pl.col("shared_kmers_241") > 0)
        ax.plot(
            pos["k"],
            pos["shared_kmers_241"],
            "-o",
            color=RAMP(0.85),
            ms=2.5,
            lw=1.2,
            zorder=3,
        )
        zero = m.filter(pl.col("shared_kmers_241") == 0)
        ax.scatter(
            zero["k"], [0.0] * zero.height, marker="x", s=6, color=NOT_FOUND, zorder=3
        )
        t = (
            theirs.filter(pl.col("alphabet") == alpha).sort("k")
            if theirs is not None
            else None
        )
        if t is not None:
            ax.scatter(
                t["k"],
                t["shared_kmers"],
                s=14,
                facecolors="none",
                edgecolors="#c0392b",
                lw=0.8,
                zorder=4,
            )
        if t is not None:
            both = t.join(m, on="k", how="inner").with_columns(
                (pl.col("shared_kmers") == pl.col("shared_kmers_241")).alias("agree")
            )
            note = (
                f"both ran k={both['k'].min()}-{both['k'].max()}:\n{both['agree'].sum()} of {both.height} counts agree"
                if both.height
                else "no k in common"
            )
            ax.text(
                0.97,
                0.97,
                note,
                transform=ax.transAxes,
                va="top",
                ha="right",
                color="#444444",
            )
        ax.set_yscale("symlog", linthresh=1)
        ax.set_ylim(-0.4, ymax * 2)
        ax.set_xlim(4, 30)
        ax.set_title(f"{alpha} ({CLASSES[alpha]})")
        ax.grid(alpha=0.25, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        if i == 0:
            ax.set_ylabel("exact k-mers P66 and CD47 share")
        else:
            plt.setp(ax.get_yticklabels(), visible=False)
        if i == n // 2:
            ax.set_xlabel("k (letters in one seed)")

    ax = axes[-1]
    if control is not None:
        c = control.group_by("k").agg(pl.col("shared_kmers").alias("counts")).sort("k")
        src = (
            theirs.filter(pl.col("alphabet") == "hp_lehninger2")
            if theirs is not None
            else mine.filter(pl.col("alphabet") == "hp_lehninger2").rename(
                {"shared_kmers_241": "shared_kmers"}
            )
        )
        cd47_k = {r["k"]: r["shared_kmers"] for r in src.iter_rows(named=True)}
        for row in c.iter_rows(named=True):
            k, counts = row["k"], np.array(row["counts"])
            cd = cd47_k.get(k)
            if cd is None:
                continue
            pct = 100.0 * (counts < cd).mean()
            ax.bar(k, pct, width=0.55, color="#d0d0d0", zorder=2)
            ax.text(k, pct + 1.5, f"{pct:.0f}%", ha="center", color="#555555")
    agg = (
        cd47.group_by("ksize")
        .agg(
            pl.col("percentile_among_hits").min().alias("lo"),
            pl.col("percentile_among_hits").max().alias("hi"),
        )
        .sort("ksize")
    )
    for row in agg.iter_rows(named=True):
        k, lo, hi = row["ksize"], row["lo"], row["hi"]
        ax.plot(
            [k, k],
            [lo, hi],
            color=RANK_RAMP(0.3),
            lw=1.2,
            zorder=3,
            solid_capstyle="butt",
        )
        ax.scatter([k, k], [lo, hi], s=8, color=RANK_RAMP(0.3), zorder=4)
    ax.set_ylim(0, 112)
    ks = (
        [k for k in list(agg["ksize"]) + list(c["k"])]
        if control is not None
        else list(agg["ksize"])
    )
    ax.set_xlim(min(ks) - 2, max(ks) + 2)
    ax.set_xticks(sorted(set(int(k) for k in ks)))
    ax.set_xlabel("k (letters in one seed)")
    ax.set_ylabel("% of other human proteins CD47 beats")
    ax.set_title("is CD47 special? (hp_lehninger2)")
    ax.grid(alpha=0.25, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    _finish(
        fig,
        path,
        tight_layout=False,
        tools=TOOLS.replace("index + search + pair", "pair (no database) and search"),
        hypothesis=hypothesis,
        conclusion=conclusion,
        header_y=1.01,
        footer_y=-0.03,
    )


def lambda_zero_figure(
    arms: pl.DataFrame, path: Path, hypothesis: str, conclusion: str
) -> pl.DataFrame:
    """Which regions can get an E-value at all.

    Left: the share of regions at each setting that get no E-value, because the region's match
    probability u is at or above C / (1 + C) and its score scale comes out as zero. u is
    the chance that one random position of the query span and one random position of the
    target span fall in the same class, from the two spans' class counts.

    Right, one bar per alphabet: the average score of one position of a chance match,
    p - C (1 - p), with the database's chance match probability p and the alphabet's
    mismatch penalty C. An E-value only exists when this is below zero. Above zero, a
    match made of nothing but chance keeps gaining score the longer it runs, so there is
    no score at which a match becomes surprising and no E-value can be computed.
    """
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            ls="",
            ms=4,
            color=RAMP(0.75),
            label="left: colour = share of the setting's regions that get no E-value",
        ),
        Line2D(
            [],
            [],
            marker="x",
            ls="",
            ms=3.5,
            color=NOT_FOUND,
            label="left: no region at this setting for any of the three queries",
        ),
        Line2D(
            [],
            [],
            marker="s",
            ls="",
            ms=4.5,
            color=RAMP(0.75),
            label="right: average score of one position of a chance match; it has to be below zero for an E-value to exist at all",
        ),
    ]
    fig, axes = _figure_with_legend_row(
        2, (183, 95), handles, sharey=True, width_ratios=[1.6, 1.0]
    )
    ax = axes[0]
    for row in arms.iter_rows(named=True):
        y, x = ROW[row["alphabet"]], row["bits"]
        f = row.get("frac_lambda_zero")
        if f is None:
            ax.scatter(x, y, s=8, marker="x", color=NOT_FOUND, zorder=3)
        else:
            ax.scatter(
                x, y, s=MARK_S, c=[RAMP(norm(f))], edgecolors="white", lw=0.4, zorder=4
            )
    ax.set_title("share of regions that get no E-value, by setting")
    _rows_axis(ax)
    cb = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=RAMP), ax=ax, fraction=0.04, pad=0.02
    )
    cb.set_label("share of the setting's regions with no E-value")

    ax = axes[1]
    per = arms.group_by("alphabet").agg(
        pl.col("chance_drift").first(),
        pl.col("p_match").first(),
        pl.col("penalty").first(),
    )
    for row in per.iter_rows(named=True):
        y = ROW[row["alphabet"]]
        d = row["chance_drift"]
        if d is None:
            continue
        ax.barh(y, d, height=0.55, color=RAMP(0.75), zorder=2)
        # Labels sit right of the zero line (or of a positive bar), never on a bar.
        ax.text(
            max(d, 0) + 0.03,
            y,
            f"p = {row['p_match']:.2f}, C = {row['penalty']:g}",
            va="center",
            ha="left",
        )
    ax.axvline(0, color="#c0392b", ls="--", lw=1.2, zorder=3)
    ax.set_xlim(-1.6, 0.9)
    ax.set_xlabel("average score of one position of a chance match:\np − C (1 − p)")
    ax.set_title("can this alphabet have an E-value at all?")
    ax.text(
        -1.55,
        -0.6,
        "< 0: E-value exists",
        ha="left",
        va="center",
    )
    ax.text(
        0.05,
        -0.6,
        "≥ 0: none",
        ha="left",
        va="center",
    )
    _row_ticks(ax, labels=False)
    ax.grid(axis="x", alpha=0.25, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    _finish(
        fig,
        path,
        tight_layout=False,
        tools=TOOLS,
        hypothesis=hypothesis,
        conclusion=conclusion,
        header_y=1.01,
        footer_y=-0.02,
    )
    return per.sort("chance_drift", descending=True)


# ---------------------------------------------------------------------------
# Every human protein's rank, not only the partner's: the BCL2 family and the
# real-protein null.
# ---------------------------------------------------------------------------

RESULTS = Path(__file__).resolve().parent / "241_results"
HUMAN_FASTA = Path(
    os.environ.get(
        "NB241_HUMAN",
        "/Users/olga/data/gencode/human/v49/gencode.v49.pc_translations.canonical.fa",
    )
)

# (column in regions.parquet, lower is better) for the five metrics of sections 3 to 5.
REGION_COLUMNS = {
    "E-value": ("region_evalue", True),
    "mean IDF": ("region_mean_idf", False),
    "tf-idf": ("region_tfidf", False),
    "enrichment": ("region_enrichment", False),
    "Poisson p-value": ("region_tail_probability", True),
}

BCL2_PFAM = "PF00452"  # Pfam Bcl-2 domain, shared by Ced-9 and human BCL2
UNIPROT_PFAM_URL = (
    "https://rest.uniprot.org/uniprotkb/search?query=xref:pfam-{pfam}+AND+"
    "organism_id:9606+AND+reviewed:true&fields=accession,gene_primary,protein_name,"
    "length,xref_pfam&format=tsv&size=500"
)


def pfam_family(pfam: str = BCL2_PFAM) -> pl.DataFrame:
    """Reviewed human UniProt entries with this Pfam domain, cached in 241_results/.

    The family is read from UniProt, not typed from memory. Delete the cache file to
    fetch it again.
    """
    cache = RESULTS / f"human_uniprot_reviewed_{pfam}.tsv"
    if not cache.exists():
        with urllib.request.urlopen(UNIPROT_PFAM_URL.format(pfam=pfam)) as r:
            text = r.read().decode()
        if not text.startswith("Entry\t"):
            raise RuntimeError(f"UniProt did not return a table: {text[:200]}")
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(text)
    return pl.read_csv(cache, separator="\t").rename(
        {
            "Entry": "accession",
            "Gene Names (primary)": "gene",
            "Protein names": "protein_name",
            "Length": "length",
            "Pfam": "pfam",
        }
    )


def all_target_ranks(
    query: str, arms: pl.DataFrame, regions_path: Path | None = None
) -> pl.DataFrame:
    """The rank of every human protein, at every setting, under each of the five metrics.

    Same rule as the collector's rank_rows: a protein's score is its best region; rank
    is 1 + the number of proteins strictly better; only proteins with a finite value are
    ranked (n_finite). E-values are left out at settings without a Karlin-Altschul fit
    and for the alphabets in no_evalue_alphabets().
    """
    regions_path = regions_path or DATA / "regions.parquet"
    bad = set(no_evalue_alphabets(arms))
    fitted = arms.filter(pl.col("fitted") == True).select(  # noqa: E712
        "alphabet", pl.col("ksize").cast(pl.Int32).alias("ksize_arm")
    )
    lf = pl.scan_parquet(regions_path).filter(pl.col("query_name") == query)
    out = []
    for metric, (col, lower) in REGION_COLUMNS.items():
        m = lf
        if metric == "E-value":
            m = m.join(fitted.lazy(), on=["alphabet", "ksize_arm"]).filter(
                ~pl.col("alphabet").is_in(list(bad))
            )
        agg = pl.col(col).min() if lower else pl.col(col).max()
        per = (
            m.group_by("alphabet", "ksize_arm", "bits", "target_name", "gene")
            .agg(agg.alias("value"))
            .filter(pl.col("value").is_not_null() & pl.col("value").is_finite())
            .with_columns(
                pl.col("value")
                .rank("min", descending=not lower)
                .over("alphabet", "ksize_arm")
                .cast(pl.Int64)
                .alias("rank"),
                pl.len().over("alphabet", "ksize_arm").alias("n_finite"),
                (pl.len().over("alphabet", "ksize_arm", "value") - 1).alias("n_tied"),
                pl.lit(metric).alias("metric"),
            )
        )
        out.append(per.collect())
    return (
        pl.concat(out)
        .rename({"ksize_arm": "ksize"})
        .with_columns(pl.col("ksize").cast(pl.Int64))
        # group_by returns rows in no fixed order; sort so the nulls draw the same
        # numbers on every run.
        .sort("alphabet", "ksize", "metric", "rank", "target_name")
    )


def best_rank_per_gene(target_ranks: pl.DataFrame, genes: list[str]) -> pl.DataFrame:
    """Each gene's best rank over every setting and metric, and where it got it.

    Ties in rank are broken by fewer tied proteins, then fewer bits, as in
    best_per_alphabet. A gene with no region anywhere has a null rank.
    """
    best = (
        target_ranks.filter(pl.col("gene").is_in(genes))
        .sort("rank", "n_tied", "bits")
        .group_by("gene")
        .first()
        .select(
            "gene", "rank", "n_tied", "n_finite", "alphabet", "ksize", "bits", "metric"
        )
    )
    n_settings = (
        target_ranks.filter(pl.col("gene").is_in(genes))
        .group_by("gene")
        .agg(pl.struct("alphabet", "ksize", "metric").n_unique().alias("n_combos_hit"))
    )
    return (
        pl.DataFrame({"gene": genes})
        .join(best, on="gene", how="left")
        .join(n_settings, on="gene", how="left")
        .with_columns((100 * pl.col("rank") / pl.col("n_finite")).alias("percent"))
        .sort("rank", nulls_last=True)
    )


def top_hits_at(
    target_ranks: pl.DataFrame, alphabet: str, k: int, metric: str, n: int = 10
) -> pl.DataFrame:
    """The n best proteins at one setting and metric, in rank order, ties by gene."""
    return (
        target_ranks.filter(
            (pl.col("alphabet") == alphabet)
            & (pl.col("ksize") == k)
            & (pl.col("metric") == metric)
            & (pl.col("rank") <= n)
        )
        .sort("rank", "gene")
        .select("rank", "n_tied", "n_finite", "gene", "value")
    )


def best_per_protein(target_ranks: pl.DataFrame, on: str = "rank") -> np.ndarray:
    """The best `on` ("rank", or "percent" = 100 * rank / n_finite) every human protein
    reaches over every setting and metric in `target_ranks`, one value per protein of
    the 19_732; inf for a protein with no region anywhere."""
    best = (
        target_ranks.with_columns(
            (100 * pl.col("rank") / pl.col("n_finite")).alias("percent")
        )
        .group_by("target_name")
        .agg(pl.col(on).min())[on]
        .to_numpy()
        .astype(float)
    )
    never = max(N_HUMAN - best.size, 0)
    return np.concatenate([best, np.full(never, np.inf)])


def best_of_many_nulls(
    target_ranks: pl.DataFrame,
    members: list[str],
    on: str = "rank",
    name: str | None = None,
    n_draws: int = 20_000,
    seed: int = 0,
) -> dict:
    """Two nulls for the best `on` (rank, or percent of the hit list) a set of proteins
    reaches over the whole sweep.

    uniform: in every setting x metric where a member has a finite value, that member's
    rank is drawn uniformly from 1..n_finite, all draws independent; keep the smallest.
    For one member it is the null of the first version of this notebook.

    real proteins: draw len(members) human proteins that are not members, take each
    one's own best over the same settings and metrics, keep the smallest. This keeps
    how one protein's ranks move together across neighbouring k and between metrics,
    which the uniform null ignores; but a random protein is hit at fewer settings than
    the partner, which was picked because it is hit.

    Both report the share of draws at least as good as the members' observed best.
    """
    rng = np.random.default_rng(seed)
    tr = target_ranks.with_columns(
        (100 * pl.col("rank") / pl.col("n_finite")).alias("percent")
    )
    mem = tr.filter(pl.col("gene").is_in(members))
    observed = float(mem[on].min())
    n = mem["n_finite"].to_numpy()
    uniform = rng.integers(1, n[None, :] + 1, size=(n_draws, n.size))
    if on == "percent":
        uniform = 100 * uniform / n[None, :]
    uniform = uniform.min(axis=1)
    others = tr.filter(~pl.col("gene").is_in(members))
    pool = np.sort(best_per_protein(others, on))[: N_HUMAN - len(members)]
    real = pool[rng.integers(0, pool.size, size=(n_draws, len(members)))].min(axis=1)
    return {
        "set": name
        or (members[0] if len(members) == 1 else f"{len(members)} proteins"),
        "on": on,
        "n_members": len(members),
        "n_member_combos": int(n.size),
        "observed_best": observed,
        "uniform_median": float(np.median(uniform)),
        "uniform_p05": float(np.percentile(uniform, 5)),
        "uniform_p95": float(np.percentile(uniform, 95)),
        "uniform_p_at_least_as_good": float((uniform <= observed).mean()),
        "real_median": float(np.median(real)),
        "real_p05": float(np.percentile(real, 5)),
        "real_p95": float(np.percentile(real, 95)),
        "real_p_at_least_as_good": float((real <= observed).mean()),
    }


NULL_KINDS = (("uniform", "uniform null"), ("real", "real proteins"))
BEATS_CHANCE_PERCENT = 5  # the decision rule: random as good in at most 5% of draws


def _null_panel(ax, rows: list[dict], labels: bool, text_x: float) -> None:
    """Grey bar = middle 90% of a null, dot = the observed best; two nulls per set.
    The share of random draws as good as the observed best is written at `text_x`, in
    one column, so it never runs into a bar."""
    ticks = []
    for i, r in enumerate(rows):
        for j, (kind, name) in enumerate(NULL_KINDS):
            y = 2.6 * i + j
            lo, hi = r[f"{kind}_p05"], r[f"{kind}_p95"]
            ax.barh(
                y,
                max(hi - lo, lo * 0.1),
                left=lo,
                height=0.5,
                color="#d0d0d0",
                zorder=2,
            )
            ax.scatter(r["observed_best"], y, s=MARK_S, color=RANK_RAMP(0.25), zorder=4)
            p = 100 * r[f"{kind}_p_at_least_as_good"]
            ax.text(text_x, y, f"{p:.1f}%", va="center", ha="right")
            ticks.append((y, f"{r['set']}, {name}"))
            ax.axhline(y, color=GUIDE, lw=0.5, zorder=0)
    ax.set_yticks([y for y, _ in ticks])
    ax.set_yticklabels([t for _, t in ticks] if labels else [])
    ax.set_ylim(2.6 * len(rows) - 1.1, -1.3)
    ax.text(
        text_x,
        -1.0,
        f"random as good\n(≤{BEATS_CHANCE_PERCENT}% beats chance)",
        va="center",
        ha="right",
    )
    ax.set_xscale("log")


def family_figure(
    family_best: pl.DataFrame,
    nulls: pl.DataFrame,
    path: Path,
    title: str,
    hypothesis: str,
    conclusion: str,
) -> None:
    """a: the best rank each BCL2-family protein reaches when Ced9 is the query.
    b, c: the best rank (b) and the best percent of the hit list (c) of BCL2 alone and
    of the whole family, against the uniform and the real-protein null. The column of
    numbers on the right of b and c is the share of random draws as good as the
    observed best; beating chance needs 5% or less."""
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            ls="",
            ms=4,
            color=RANK_RAMP(0.25),
            label="observed: the best over every setting and the five metrics",
        ),
        Line2D(
            [], [], marker="s", ls="", ms=4.5, color="#f0dcae", label="a: the top 10"
        ),
        Line2D(
            [],
            [],
            marker="s",
            ls="",
            ms=4.5,
            color="#d0d0d0",
            label="b, c: middle 90% of the same best for as many random human proteins (20_000 draws)",
        ),
        Line2D(
            [],
            [],
            ls="",
            label=f"b, c, right column: % of random draws as good or better; "
            f"beating chance needs {BEATS_CHANCE_PERCENT}% or less",
        ),
    ]
    w_in, h_in = 183 * pf.MM, 120 * pf.MM
    fig = plt.figure(figsize=(w_in, h_in))
    band_in = len(handles) * 6 * 1.6 / 72 + 0.03
    # Two levels of spacing: the legend band sits right on top of the panels, while the
    # panels keep room between them for panel a's x label and the b, c titles.
    top_in = 0.995 * h_in
    outer = fig.add_gridspec(
        2,
        1,
        height_ratios=[band_in, h_in - band_in - 0.45],
        left=1.45 / w_in,
        right=0.99,
        top=0.995,
        bottom=0.45 / h_in,
        hspace=0.18 / (top_in / 2),
    )
    gs = outer[1].subgridspec(
        2,
        2,
        height_ratios=[2.3, 1.45],
        width_ratios=[1.0, 0.8],
        hspace=0.55,
        wspace=0.12,
    )
    lax = fig.add_subplot(outer[0])
    lax.axis("off")
    lax.legend(
        handles=handles,
        loc="upper left",
        frameon=False,
        borderaxespad=0,
        borderpad=0,
        handletextpad=0.4,
        labelspacing=0.35,
    )
    ax = fig.add_subplot(gs[0, :])
    ax.axvspan(1, 10, color="#f0dcae", zorder=1)
    for i, row in enumerate(family_best.iter_rows(named=True)):
        ax.axhline(i, color=GUIDE, lw=0.5, zorder=0)
        if row["rank"] is None:
            ax.scatter(1.3, i, marker="x", s=8, color=NOT_FOUND, zorder=3)
            continue
        ax.scatter(row["rank"], i, s=MARK_S, color=RANK_RAMP(0.25), zorder=4)
        ax.text(
            row["rank"] * 1.4,
            i,
            f"{row['rank']:_} of {row['n_finite']:_}  ({row['alphabet']} k={row['ksize']}, {row['metric']})",
            va="center",
        )
    ax.set_yticks(range(family_best.height))
    ax.set_yticklabels(family_best["gene"])
    ax.set_ylim(family_best.height - 0.4, -0.7)
    ax.set_xscale("log")
    ax.set_xlim(0.8, 3e5)
    ax.set_xticks([1, 10, 100, 1000])
    ax.set_xticklabels(["1", "10", "100", "1_000"])
    ax.set_xlabel("best rank among the human proteins ranked, Ced9 query (1 = best)")
    ax.set_title(
        f"a  every human protein with the Bcl-2 domain ({BCL2_PFAM})",
        loc="left",
        fontweight="bold",
    )

    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])
    for ax_, on, xlabel, labels, text_x, panel_title in (
        (
            ax_b,
            "rank",
            "best rank over the sweep (1 = best)",
            True,
            3e5,
            "b  best rank, against random proteins",
        ),
        (
            ax_c,
            "percent",
            "best rank / proteins ranked (%)",
            False,
            3e3,
            "c  best % of the list, against random proteins",
        ),
    ):
        _null_panel(
            ax_,
            [r for r in nulls.iter_rows(named=True) if r["on"] == on],
            labels,
            text_x,
        )
        ax_.set_xlabel(xlabel)
        ax_.set_title(panel_title, loc="left", fontweight="bold")
    ax_b.set_xlim(0.7, 3e5)
    ax_b.set_xticks([1, 10, 100, 1000])
    ax_b.set_xticklabels(["1", "10", "100", "1_000"])
    ax_c.set_xlim(3e-3, 3e3)
    ax_c.set_xticks([0.01, 0.1, 1, 10])
    ax_c.set_xticklabels(["0.01", "0.1", "1", "10"])
    _finish(
        fig,
        path,
        tight_layout=False,
        tools=TOOLS,
        title=title,
        hypothesis=hypothesis,
        conclusion=conclusion,
        header_y=1.01,
        footer_y=-0.02,
    )


# ---------------------------------------------------------------------------
# One region, residue by residue.
# ---------------------------------------------------------------------------


def read_fasta(path: Path) -> dict[str, str]:
    seqs: dict[str, list[str]] = {}
    name = None
    for line in Path(path).read_text().splitlines():
        if line.startswith(">"):
            name = line[1:].split()[0]
            seqs[name] = []
        elif name is not None:
            seqs[name].append(line.strip())
    return {k: "".join(v) for k, v in seqs.items()}


def class_string(seq: str, alphabet: str) -> str:
    """The sequence in the alphabet's classes. Two-class HP alphabets print H
    (hydrophobic, the first class) and P (polar); others print a, b, c, ... in the
    order of ALPHABET_CLUSTERS."""
    clusters = ALPHABET_CLUSTERS[alphabet]
    letters = (
        "HP"
        if len(clusters) == 2 and alphabet.startswith("hp_")
        else "abcdefghijklmnopqrst"
    )
    code = {r: letters[i] for i, cl in enumerate(clusters) for r in cl}
    return "".join(code.get(r, "?") for r in seq.upper())


def region_residues(
    region: dict, query_seq: str, target_seq: str, alphabet: str
) -> str:
    """The query and target spans of one gapless region, their classes, and match lines
    (| = same residue, or same class on the class lines). Coordinates are 0-based, end
    excluded, as kmerseek writes them."""
    qs, qe = int(region["region_start"]), int(region["region_end"])
    ts, te = int(region["target_start"]), int(region["target_end"])
    q, t = query_seq[qs:qe], target_seq[ts:te]
    qc, tc = class_string(q, alphabet), class_string(t, alphabet)
    same = "".join("|" if a == b else " " for a, b in zip(q, t))
    same_c = "".join("|" if a == b else " " for a, b in zip(qc, tc))
    w = max(len(region["query_name"]), len(region["gene"]))
    return "\n".join(
        [
            f"{region['query_name']:<{w}}  {qs:>4} {q} {qe}",
            f"{'':<{w}}       {same}",
            f"{region['gene']:<{w}}  {ts:>4} {t} {te}",
            "",
            f"{region['query_name']:<{w}}  {qs:>4} {qc} {qe}",
            f"{'':<{w}}       {same_c}",
            f"{region['gene']:<{w}}  {ts:>4} {tc} {te}",
            "",
            f"{len(q)} residues, {sum(a == b for a, b in zip(q, t))} identical, "
            f"{sum(a == b for a, b in zip(qc, tc))} in the same class "
            f"({len(q) - sum(a == b for a, b in zip(qc, tc))} class mismatches)",
        ]
    )
