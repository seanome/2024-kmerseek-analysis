"""Figures for the three-case alphabet sweep (notebook 241).

Data comes from 241_alphabet_ranking_collect.py: arms.csv and ranks.csv under
/Users/olga/data/botryllus/alphabet-ranking-three-cases/.

Most figures are a dot matrix: one row per alphabet, seed information in bits along x,
one dot per arm. Rows are grouped by how many letters the alphabet has, with a blank
row between groups, and a faint guide line runs through each row at the same height as
that alphabet's tick, so a dot can be traced back to its name.

Marks mean the same thing in every figure here:

* a filled dot, coloured on one purple ramp, is the quantity that figure is about;
* a grey cross means there is nothing at that arm (no region, or the arm was not run);
* a small black dot means the arm has no E-value, because the Karlin-Altschul fit was
  refused;
* a black ring around a filled dot marks the one arm feature named in that figure's
  legend.

The legend sits above the panels, in a band only as tall as the legend itself.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.lines import Line2D

from hp_conservation_utils import finish_figure  # noqa: F401  (re-exported)

DATA = Path("/Users/olga/data/botryllus/alphabet-ranking-three-cases")
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
NO_EVALUE_MARKER = dict(marker=".", s=26, color=NO_EVALUE, lw=0)
GUIDE = "#ececec"


def _no_evalue_handle(label: str) -> Line2D:
    return Line2D([], [], marker=".", ls="", ms=9, color=NO_EVALUE, label=label)


def load() -> tuple[pl.DataFrame, pl.DataFrame]:
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
        },
    )
    return arms, ranks


def _figure_with_legend_row(
    ncols: int,
    figsize: tuple,
    handles: list,
    sharey: bool = True,
    width_ratios=None,
    legend_fontsize: float = 9.0,
    title_in: float = 0.32,
    xlabel_in: bool = True,
):
    """A figure whose top row is an axes holding only the legend, so the legend reads
    before the marks. The band is only as tall as the legend text, so there is no gap
    between the TOOLS line above the figure and the legend."""
    w_in, h_in = figsize
    fig = plt.figure(figsize=figsize)
    n_lines = len(handles)
    # The legend band is only as tall as the legend text, and the panels start right
    # under it. Margins are set here rather than left to tight_layout, which refuses to
    # run on this layout and would leave matplotlib's default 0.88 top margin: a band of
    # white between the TOOLS line above the figure and the legend.
    band_in = n_lines * legend_fontsize * 1.75 / 72.0 + 0.06
    top, bottom = 0.995, min(0.25, (0.62 + 0.20 * bool(xlabel_in)) / h_in)
    left, right = min(0.30, 1.55 / w_in), 0.995
    gap = title_in / h_in
    span = top - bottom
    gs = fig.add_gridspec(
        2,
        ncols,
        height_ratios=[band_in, max(h_in - band_in, 1.0)],
        width_ratios=width_ratios,
        top=top,
        bottom=bottom,
        left=left,
        right=right,
        hspace=2 * gap / max(span - gap, 0.05),
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
        handletextpad=0.6,
        labelspacing=0.35,
        borderpad=0.0,
    )
    axes = []
    for i in range(ncols):
        ax = fig.add_subplot(gs[1, i], sharey=axes[0] if (axes and sharey) else None)
        if axes and sharey:
            plt.setp(ax.get_yticklabels(), visible=False)
        axes.append(ax)
    return fig, axes


def _row_ticks(ax, labels: bool = True):
    """The y axis of an alphabet-row panel: one tick per alphabet, blank rows between
    the letter-count groups, and a guide line at the height of each tick."""
    ax.set_yticks([ROW[a] for a in ORDER])
    if labels:
        ax.set_yticklabels([f"{a} ({CLASSES[a]})" for a in ORDER], fontsize=8)
    ax.set_ylim(Y_LAST + 0.9, -0.9)
    for a in ORDER:
        ax.axhline(ROW[a], color=GUIDE, lw=0.8, zorder=0)


def _rows_axis(ax, xlabel: bool = True):
    _row_ticks(ax)
    ax.set_xlim(13, 47)
    ax.set_xticks([16, 20, 24, 28, 32, 36, 40, 44])
    if xlabel:
        ax.set_xlabel(
            "information in one seed (bits = k × bits per position)", fontsize=9
        )
    ax.grid(axis="x", alpha=0.25, zorder=0)
    ax.tick_params(axis="x", labelsize=8)
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
            ms=8,
            color=RAMP(0.85),
            label="the Karlin-Altschul fit worked: this arm has E-values",
        ),
        _no_evalue_handle(
            "the fit was refused even after the retry: this arm has every metric except the E-value"
        ),
    ]
    if arms.filter(pl.col("fitted").is_null()).height:
        handles.append(
            Line2D(
                [],
                [],
                marker="x",
                ls="",
                ms=7,
                color=NOT_FOUND,
                label="the arm was not run",
            )
        )
    fig, axes = _figure_with_legend_row(1, (11, 7.0), handles)
    ax = axes[0]
    for row in arms.iter_rows(named=True):
        y, x = ROW[row["alphabet"]], row["bits"]
        if row["fitted"] is True:
            ax.scatter(
                x, y, s=62, color=RAMP(0.85), edgecolors="white", lw=0.4, zorder=4
            )
        elif row["fitted"] is False:
            ax.scatter(x, y, zorder=4, **NO_EVALUE_MARKER)
        else:
            ax.scatter(x, y, s=30, marker="x", color=NOT_FOUND, lw=1.0, zorder=3)
    ax.set_title("which arms have an E-value", fontsize=11)
    _rows_axis(ax)
    finish_figure(
        fig,
        path,
        tight=False,
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
            pl.len().alias("arms"),
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
            ms=8,
            color=RANK_RAMP(0.15),
            label=f"{partner} is among the hits; colour = its rank, number written when rank ≤ {annotate_le}",
        ),
        Line2D(
            [],
            [],
            marker="x",
            ls="",
            ms=7,
            color=NOT_FOUND,
            label=f"{partner} is not among the hits at this arm",
        ),
        _no_evalue_handle(
            "E-value panel only: this arm has no E-value (the Karlin-Altschul fit was refused)"
        ),
    ]
    fig, axes = _figure_with_legend_row(5, (21, 8.0), handles)
    for ax, metric in zip(axes, METRICS5):
        m = sub.filter(pl.col("metric") == metric)
        for row in m.iter_rows(named=True):
            y = ROW[row["alphabet"]]
            x = row["bits"]
            if not row["searched"]:
                continue
            if metric == "E-value" and row["fitted"] is not True:
                ax.scatter(x, y, zorder=3, **NO_EVALUE_MARKER)
                continue
            if not row["partner_found"]:
                ax.scatter(x, y, s=30, marker="x", color=NOT_FOUND, lw=1.0, zorder=3)
                continue
            r = row["rank"]
            ax.scatter(
                x, y, s=62, c=[RANK_RAMP(norm(r))], edgecolors="white", lw=0.4, zorder=4
            )
            if r <= annotate_le:
                ax.text(
                    x,
                    y,
                    str(r),
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color="white",
                    zorder=5,
                    fontweight="bold",
                )
        ax.set_title(metric, fontsize=11)
        _rows_axis(ax, xlabel=(metric == METRICS5[2]))

    cb = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=RANK_RAMP),
        ax=axes,
        fraction=0.012,
        pad=0.01,
        aspect=30,
    )
    cb.set_label(
        f"rank of {partner} among the human proteins hit (1 = best)", fontsize=9
    )
    cb.set_ticks([1, 10, 100, 1000, 10_000])
    cb.set_ticklabels(["1", "10", "100", "1_000", "10_000"])

    finish_figure(
        fig,
        path,
        tight=False,
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
    of the five metrics, and which arm and metric got there."""
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
    """Two panels: the best rank BCL2 (gold standard) and CD47 (the reach) ever reach,
    per alphabet, over every k and metric."""
    handles = [
        Line2D(
            [],
            [],
            marker="s",
            ls="",
            ms=9,
            color=RANK_RAMP(0.5),
            label="bar: best rank reached, over every k and every one of the five metrics; label says which",
        ),
        Line2D(
            [],
            [],
            marker="x",
            ls="",
            ms=7,
            color=NOT_FOUND,
            label="the partner is never among the hits for this alphabet at any k",
        ),
    ]
    fig, axes = _figure_with_legend_row(2, (13, 7.0), handles)
    tables = []
    for ax, (q, partner, label) in zip(
        axes,
        [
            ("Ced9", "BCL2", "gold standard: Ced9 query, BCL2 is the known homolog"),
            ("P66", "CD47", "the reach: P66 query, CD47 is the proposed partner"),
        ],
    ):
        b = best_per_alphabet(ranks, q)
        tables.append(b.with_columns(pl.lit(q).alias("query")))
        for row in b.iter_rows(named=True):
            y = ROW[row["alphabet"]]
            if row["rank"] is None:
                ax.scatter(1.0, y, marker="x", color=NOT_FOUND, s=40, zorder=3)
                ax.text(
                    1.25,
                    y,
                    "never among the hits",
                    va="center",
                    fontsize=7,
                    color=NOT_FOUND,
                )
                continue
            r = row["rank"]
            ax.barh(y, r, left=1, height=0.55, color=RANK_RAMP(0.5), zorder=2)
            ax.text(
                r * 1.12,
                y,
                f"{r:_} of {row['n_targets']:_}  ({row['metric']}, k={row['ksize']}"
                + (f", {row['n_tied']} tied)" if row["n_tied"] else ")"),
                va="center",
                fontsize=6.8,
            )
        ax.set_xscale("log")
        ax.set_xlim(1, 2e7)
        ax.set_xticks([1, 10, 100, 1000, 10_000])
        ax.set_xticklabels(["1", "10", "100", "1_000", "10_000"])
        ax.set_xlabel(
            f"best rank of {partner} among the human proteins hit (1 = best)",
            fontsize=9,
        )
        ax.set_title(label, fontsize=10.5)
        _row_ticks(ax)
        ax.grid(axis="x", alpha=0.25, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
    finish_figure(
        fig,
        path,
        tight=False,
        tools=TOOLS,
        hypothesis=hypothesis,
        conclusion=conclusion,
        header_y=1.01,
        footer_y=-0.02,
    )
    return pl.concat(tables)


def random_protein_null(
    ranks: pl.DataFrame, query: str, n_draws: int = 20_000, seed: int = 0
) -> dict:
    """How well a human protein picked at random from the same hit lists would do.

    The partner's headline number is the best rank it reaches anywhere in the sweep, so
    it is the smallest of many ranks. The fair comparison is the smallest of the same
    many ranks for a protein drawn at random from each arm's hit list: a protein that is
    among the n hits of an arm has rank uniform on 1..n under that null.
    """
    sub = ranks.filter(
        (pl.col("query") == query)
        & pl.col("metric").is_in(METRICS5)
        & pl.col("partner_found")
        & pl.col("rank").is_not_null()
    )
    n = sub["n_targets"].to_numpy()
    observed = int(sub["rank"].min())
    rng = np.random.default_rng(seed)
    draws = rng.integers(1, n[None, :] + 1, size=(n_draws, len(n))).min(axis=1)
    return {
        "query": query,
        "n_combos": len(n),
        "observed_best_rank": observed,
        "n_targets_at_best": int(
            sub.filter(pl.col("rank") == observed)["n_targets"][0]
        ),
        "null_median": float(np.median(draws)),
        "null_p05": float(np.percentile(draws, 5)),
        "null_p95": float(np.percentile(draws, 95)),
        "p_random_at_least_as_good": float((draws <= observed).mean()),
    }


def null_rank_figure(
    ranks: pl.DataFrame, path: Path, hypothesis: str, conclusion: str
) -> pl.DataFrame:
    """One panel: the best rank the known partner reaches anywhere in the sweep, against
    the best rank a human protein drawn at random from the same hit lists reaches."""
    rows = [random_protein_null(ranks, q) for q in ("Ced9", "P66")]
    partners = {"Ced9": "BCL2", "P66": "CD47"}
    handles = [
        Line2D(
            [],
            [],
            marker="s",
            ls="",
            ms=9,
            color="#d9b25a",
            label="the top 10 of the hit list: what a search has to reach to be usable",
        ),
        Line2D(
            [],
            [],
            marker="s",
            ls="",
            ms=9,
            color="#d0d0d0",
            label="best rank of a human protein drawn at random from the same hit lists, middle 90% of 20_000 draws",
        ),
        Line2D(
            [],
            [],
            marker="o",
            ls="",
            ms=9,
            color=RANK_RAMP(0.25),
            label="best rank of the known partner, over 19 alphabets, every k and all five metrics",
        ),
    ]
    fig, axes = _figure_with_legend_row(1, (11, 4.2), handles)
    ax = axes[0]
    ax.axvspan(1, 10, color="#f0dcae", zorder=1)
    ax.text(10, -0.70, " top 10", fontsize=8.5, color="#8a6d1f", va="center", ha="left")
    for i, r in enumerate(rows):
        ax.barh(
            i,
            r["null_p95"] - r["null_p05"],
            left=r["null_p05"],
            height=0.34,
            color="#d0d0d0",
            zorder=2,
        )
        ax.scatter(
            r["observed_best_rank"],
            i,
            s=110,
            color=RANK_RAMP(0.25),
            edgecolors="white",
            lw=0.8,
            zorder=4,
        )
        ax.text(
            r["observed_best_rank"] * 1.15,
            i - 0.02,
            f"  {r['observed_best_rank']:_} of {r['n_targets_at_best']:_} hit",
            va="center",
            fontsize=9,
        )
        ax.text(
            11.5,
            i + 0.33,
            f"a human protein drawn at random does as well or better in "
            f"{100 * r['p_random_at_least_as_good']:.0f}% of draws "
            f"({r['n_combos']} arm × metric combinations)",
            va="center",
            fontsize=8.5,
            color="#444444",
        )
    ax.set_yticks([0, 1])
    ax.set_yticklabels(
        [
            f"Ced9 → {partners['Ced9']}\nknown homolog",
            f"P66 → {partners['P66']}\nnot yet shown",
        ],
        fontsize=9,
    )
    ax.set_ylim(1.75, -0.85)
    ax.set_xscale("log")
    ax.set_xlim(1, 30_000)
    ax.set_xticks([1, 10, 100, 1000, 10_000])
    ax.set_xticklabels(["1", "10", "100", "1_000", "10_000"])
    ax.set_xlabel(
        "rank among the human proteins the search hit (1 = top of the list)", fontsize=9
    )
    ax.grid(axis="x", alpha=0.25, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    finish_figure(
        fig,
        path,
        tight=False,
        tools=TOOLS,
        hypothesis=hypothesis,
        conclusion=conclusion,
        header_y=1.01,
        footer_y=-0.04,
    )
    return pl.DataFrame(rows)


def best_per_metric(ranks: pl.DataFrame, query: str) -> pl.DataFrame:
    """For every metric the search writes, the best the partner ever does, as a percent
    of the proteins that arm hit. A rank of 213 out of 18_064 and one of 173 out of
    1_161 are not the same result, so the comparison is on the percent, not the rank."""
    return (
        ranks.filter(
            (pl.col("query") == query)
            & pl.col("partner_found")
            & pl.col("rank").is_not_null()
        )
        .with_columns(
            (100 * pl.col("rank") / pl.col("n_targets")).alias("best_percent")
        )
        .sort("best_percent", "rank")
        .group_by("metric")
        .first()
        .select(
            "metric", "best_percent", "rank", "n_targets", "n_tied", "alphabet", "ksize"
        )
        .sort("best_percent")
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
            ms=9,
            color=RANK_RAMP(0.25),
            label="best the partner ever does under this metric, as a percent of the proteins that arm hit",
        ),
        Line2D(
            [],
            [],
            marker="o",
            ls="",
            ms=9,
            markerfacecolor=RANK_RAMP(0.25),
            markeredgecolor="black",
            markeredgewidth=1.4,
            label="that best rank is a tie: other proteins have exactly the same value, and the order among them is arbitrary",
        ),
        Line2D(
            [],
            [],
            marker="s",
            ls="",
            ms=9,
            color="#d9b25a",
            label=f"where the top 10 of the human proteome would sit ({top10_pct:.2f}%)",
        ),
    ]
    fig, axes = _figure_with_legend_row(2, (15, 5.4), handles, sharey=False)
    tables = []
    for ax, (q, partner) in zip(axes, [("Ced9", "BCL2"), ("P66", "CD47")]):
        b = best_per_metric(ranks, q)
        tables.append(b.with_columns(pl.lit(q).alias("query")))
        ax.axvline(top10_pct, color="#d9b25a", lw=2.5, zorder=1)
        for i, row in enumerate(b.iter_rows(named=True)):
            tied = (row["n_tied"] or 0) > 0
            ax.scatter(
                row["best_percent"],
                i,
                s=110,
                color=RANK_RAMP(0.25),
                edgecolors="black" if tied else "white",
                lw=1.4 if tied else 0.8,
                zorder=4,
            )
            label = (
                f"  {row['rank']:_} of {row['n_targets']:_} ({row['alphabet']} k={row['ksize']}"
                + (f", {row['n_tied']:_} tied)" if tied else ")")
            )
            ax.text(row["best_percent"] * 1.2, i, label, va="center", fontsize=7.5)
        ax.set_yticks(range(b.height))
        ax.set_yticklabels(b["metric"], fontsize=8.5)
        ax.set_ylim(b.height - 0.4, -0.7)
        ax.set_xscale("log")
        ax.set_xlim(0.03, 4_000)
        ax.set_xticks([0.05, 0.1, 1, 10, 100])
        ax.set_xticklabels(["0.05", "0.1", "1", "10", "100"])
        ax.set_xlabel(
            f"best position of {partner} in the hit list (percent; smaller is better)",
            fontsize=9,
        )
        ax.set_title(f"{q} → {partner}", fontsize=10.5)
        ax.grid(axis="x", alpha=0.25, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        for y in range(b.height):
            ax.axhline(y, color=GUIDE, lw=0.8, zorder=0)
    finish_figure(
        fig,
        path,
        tight=False,
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
            ms=8,
            color=RAMP(0.75),
            label="filled dot: colour = the panel's quantity",
        ),
        Line2D(
            [],
            [],
            marker="o",
            ls="",
            ms=8,
            markerfacecolor=ramp_e(0.1),
            markeredgecolor="black",
            markeredgewidth=1.2,
            label="black ring, left panel only: best E-value below 1, gene and E written beside it",
        ),
        Line2D(
            [],
            [],
            marker="x",
            ls="",
            ms=7,
            color=NOT_FOUND,
            label="no human protein has a region at this arm",
        ),
        _no_evalue_handle(
            "left panel only: this arm has no E-value (the Karlin-Altschul fit was refused)"
        ),
    ]
    fig, axes = _figure_with_legend_row(2, (14, 8.0), handles)
    ax = axes[0]
    below_one: dict[float, list[str]] = {}
    for row in e.iter_rows(named=True):
        y, x = ROW[row["alphabet"]], row["bits"]
        if not row["searched"]:
            continue
        if row["fitted"] is not True:
            ax.scatter(x, y, zorder=3, **NO_EVALUE_MARKER)
            continue
        if row["best_value"] is None:
            ax.scatter(x, y, s=30, marker="x", color=NOT_FOUND, zorder=3)
            continue
        v = row["best_value"]
        ax.scatter(
            x,
            y,
            s=62,
            c=[ramp_e(norm_e(v))],
            edgecolors="black" if v < 1 else "white",
            lw=1.2 if v < 1 else 0.4,
            zorder=4,
        )
        if v < 1:
            below_one.setdefault(y, []).append(
                f"{row['top_gene']} E={v:.2g} at {x:.0f} bits"
            )
    # Names go in the empty space right of the last arm, with a leader from the row.
    for y, labels in below_one.items():
        ax.text(
            51.5,
            y,
            "; ".join(labels),
            va="center",
            ha="right",
            fontsize=6.5,
            zorder=5,
            bbox=dict(
                boxstyle="round,pad=0.15",
                facecolor="white",
                edgecolor="none",
                alpha=0.85,
            ),
        )
    ax.set_title("best E-value of any human protein", fontsize=11)
    _rows_axis(ax)
    ax.set_xlim(13, 52)
    cb = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm_e, cmap=ramp_e), ax=ax, fraction=0.04, pad=0.02
    )
    cb.set_label("best E-value (lower is more surprising)", fontsize=8)

    ax = axes[1]
    norm_n = mpl.colors.LogNorm(vmin=1, vmax=20_000)
    for row in e.iter_rows(named=True):
        y, x = ROW[row["alphabet"]], row["bits"]
        if not row["searched"]:
            continue
        n = row["n_targets_BHF"] or 0
        if n == 0:
            ax.scatter(x, y, s=30, marker="x", color=NOT_FOUND, zorder=3)
        else:
            ax.scatter(
                x, y, s=62, c=[RAMP(norm_n(n))], edgecolors="white", lw=0.4, zorder=4
            )
    ax.set_title("human proteins with at least one matched region", fontsize=11)
    _rows_axis(ax)
    cb = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm_n, cmap=RAMP), ax=ax, fraction=0.04, pad=0.02
    )
    cb.set_label("number of human proteins hit (of 19_732)", fontsize=8)

    finish_figure(
        fig,
        path,
        tight=False,
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
            ms=8,
            color=RAMP(0.75),
            label="the two proteins share at least one exact k-mer; colour = how many",
        ),
        Line2D(
            [],
            [],
            marker="o",
            ls="",
            ms=8,
            markerfacecolor=RAMP(0.75),
            markeredgecolor="black",
            markeredgewidth=1.2,
            label="Ced9 panel only: a shared region lies in the BH3-binding groove (Ced9 150-195, BCL2 125-170)",
        ),
        Line2D(
            [],
            [],
            marker="x",
            ls="",
            ms=7,
            color=NOT_FOUND,
            label="no shared k-mer at this arm",
        ),
    ]
    fig, axes = _figure_with_legend_row(2, (13, 8.0), handles)
    for ax, (q, partner) in zip(axes, [("Ced9", "BCL2"), ("P66", "CD47")]):
        col = f"pair_{q}_shared_kmers"
        for row in arms.iter_rows(named=True):
            y, x = ROW[row["alphabet"]], row["bits"]
            n = row[col]
            if n is None:
                continue
            if n == 0:
                ax.scatter(x, y, s=30, marker="x", color=NOT_FOUND, zorder=3)
                continue
            in_win = row.get("pair_Ced9_in_window") if q == "Ced9" else None
            ax.scatter(
                x,
                y,
                s=62,
                c=[RAMP(norm(n))],
                edgecolors="black" if in_win else "white",
                lw=1.2 if in_win else 0.4,
                zorder=4,
            )
            if n <= 9:
                ax.text(
                    x,
                    y,
                    str(n),
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color="white" if n >= 4 else "#1a1a1a",
                    zorder=5,
                    fontweight="bold",
                )
        ax.set_title(
            f"{q} vs {partner}: exact k-mers the two proteins share", fontsize=10.5
        )
        _rows_axis(ax)
    cb = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=RAMP), ax=axes, fraction=0.02, pad=0.02
    )
    cb.set_label("shared k-mers (number written when ≤ 9)", fontsize=8)
    finish_figure(
        fig,
        path,
        tight=False,
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
    get at every arm where they are among the hits and an E-value exists. The best one
    per alphabet is written at the right edge."""
    norm = mpl.colors.LogNorm(vmin=1e-1, vmax=1e6)
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            ls="",
            ms=8,
            color=RANK_RAMP(0.5),
            label="the partner is among the hits and has an E-value; colour = that E-value (dark = below 1)",
        ),
        Line2D(
            [],
            [],
            marker="x",
            ls="",
            ms=7,
            color=NOT_FOUND,
            label="the partner is not among the hits at this arm, or the arm has no E-value",
        ),
    ]
    fig, axes = _figure_with_legend_row(2, (13.5, 8.0), handles)
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
                and row["partner_found"]
                and row["partner_value"] is not None
            )
            if not ok:
                ax.scatter(x, y, s=30, marker="x", color=NOT_FOUND, zorder=3)
                continue
            v = row["partner_value"]
            ax.scatter(
                x, y, s=62, c=[RANK_RAMP(norm(v))], edgecolors="white", lw=0.4, zorder=4
            )
            best[y] = min(best.get(y, float("inf")), v)
        for y, v in best.items():
            ax.text(
                51.5,
                y,
                f"best E = {v:_.0f}" if v >= 10 else f"best E = {v:.2g}",
                va="center",
                ha="right",
                fontsize=7.5,
            )
        ax.set_title(f"{partner}'s own E-value when {q} is the query", fontsize=10.5)
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
        fontsize=8,
    )
    finish_figure(
        fig,
        path,
        tight=False,
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
            ms=7,
            color=RAMP(0.85),
            label="this sweep (notebook 241), kmerseek pair on the same two proteins",
        ),
        Line2D(
            [],
            [],
            marker="s",
            ls="",
            ms=9,
            markerfacecolor="none",
            markeredgecolor="#c0392b",
            markeredgewidth=1.4,
            label="PR #44's ladder, the same measurement run independently; it ran k values this sweep does not",
        ),
        Line2D(
            [],
            [],
            marker="x",
            ls="",
            ms=7,
            color=NOT_FOUND,
            label="this sweep ran this k and the two proteins share no k-mer",
        ),
        Line2D(
            [],
            [],
            marker="s",
            ls="",
            ms=9,
            color="#d0d0d0",
            label="right panel: percent of PR #44's 300 random human proteins that CD47 beats on shared k-mers",
        ),
        Line2D(
            [],
            [],
            marker="o",
            ls="",
            ms=8,
            color=RANK_RAMP(0.3),
            label="right panel: percent of this sweep's 19_732-protein hit list that CD47 beats, range over the five metrics",
        ),
    ]
    n = len(ladder_alphabets)
    fig, axes = _figure_with_legend_row(
        n + 1, (16, 5.6), handles, sharey=False, width_ratios=[1.0] * n + [1.7]
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
            ms=5,
            lw=1.2,
            zorder=3,
        )
        zero = m.filter(pl.col("shared_kmers_241") == 0)
        ax.scatter(
            zero["k"], [0.0] * zero.height, marker="x", s=26, color=NOT_FOUND, zorder=3
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
                s=70,
                facecolors="none",
                edgecolors="#c0392b",
                lw=1.4,
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
                fontsize=7.5,
                va="top",
                ha="right",
                color="#444444",
            )
        ax.set_yscale("symlog", linthresh=1)
        ax.set_ylim(-0.4, ymax * 2)
        ax.set_xlim(4, 30)
        ax.set_title(f"{alpha} ({CLASSES[alpha]})", fontsize=10)
        ax.grid(alpha=0.25, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)
        if i == 0:
            ax.set_ylabel("exact k-mers P66 and CD47 share", fontsize=9)
        else:
            plt.setp(ax.get_yticklabels(), visible=False)
        if i == n // 2:
            ax.set_xlabel("k (letters in one seed)", fontsize=9)

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
            ax.text(
                k, pct + 1.5, f"{pct:.0f}%", ha="center", fontsize=8, color="#555555"
            )
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
            lw=2.0,
            zorder=3,
            solid_capstyle="butt",
        )
        ax.scatter([k, k], [lo, hi], s=34, color=RANK_RAMP(0.3), zorder=4)
    ax.set_ylim(0, 112)
    ks = (
        [k for k in list(agg["ksize"]) + list(c["k"])]
        if control is not None
        else list(agg["ksize"])
    )
    ax.set_xlim(min(ks) - 2, max(ks) + 2)
    ax.set_xticks(sorted(set(int(k) for k in ks)))
    ax.set_xlabel("k (letters in one seed)", fontsize=9)
    ax.set_ylabel("percent of other human proteins CD47 beats", fontsize=9)
    ax.set_title(
        "how special is CD47 among human proteins? (hp_lehninger2)", fontsize=10
    )
    ax.grid(alpha=0.25, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)
    finish_figure(
        fig,
        path,
        tight=False,
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

    Left: the share of regions at each arm that get no E-value, because the region's own
    identity is above C / (1 + C) and its score scale comes out as zero.

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
            ms=8,
            color=RAMP(0.75),
            label="left: colour = share of the arm's regions that get no E-value",
        ),
        Line2D(
            [],
            [],
            marker="x",
            ls="",
            ms=7,
            color=NOT_FOUND,
            label="left: no region at this arm for any of the three queries",
        ),
        Line2D(
            [],
            [],
            marker="s",
            ls="",
            ms=9,
            color=RAMP(0.75),
            label="right: average score of one position of a chance match; it has to be below zero for an E-value to exist at all",
        ),
    ]
    fig, axes = _figure_with_legend_row(
        2, (14, 8.0), handles, sharey=True, width_ratios=[1.6, 1.0]
    )
    ax = axes[0]
    for row in arms.iter_rows(named=True):
        y, x = ROW[row["alphabet"]], row["bits"]
        f = row.get("frac_lambda_zero")
        if f is None:
            ax.scatter(x, y, s=30, marker="x", color=NOT_FOUND, zorder=3)
        else:
            ax.scatter(
                x, y, s=62, c=[RAMP(norm(f))], edgecolors="white", lw=0.4, zorder=4
            )
    ax.set_title("share of regions that get no E-value, by arm", fontsize=11)
    _rows_axis(ax)
    cb = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=RAMP), ax=ax, fraction=0.04, pad=0.02
    )
    cb.set_label("share of the arm's regions with no E-value", fontsize=8)

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
            fontsize=7,
        )
    ax.axvline(0, color="#c0392b", ls="--", lw=1.2, zorder=3)
    ax.set_xlim(-1.6, 0.9)
    ax.set_xlabel(
        "average score of one position of a chance match:\np − C (1 − p)", fontsize=9
    )
    ax.set_title("can this alphabet have an E-value at all?", fontsize=11)
    ax.text(
        -1.55,
        -0.6,
        "below zero: an E-value exists",
        fontsize=8,
        color="#2c6e49",
        ha="left",
        va="center",
    )
    ax.text(
        0.05,
        -0.6,
        "at or above zero: no E-value",
        fontsize=8,
        color="#c0392b",
        ha="left",
        va="center",
    )
    _row_ticks(ax, labels=False)
    ax.grid(axis="x", alpha=0.25, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    finish_figure(
        fig,
        path,
        tight=False,
        tools=TOOLS,
        hypothesis=hypothesis,
        conclusion=conclusion,
        header_y=1.01,
        footer_y=-0.02,
    )
    return per.sort("chance_drift", descending=True)
