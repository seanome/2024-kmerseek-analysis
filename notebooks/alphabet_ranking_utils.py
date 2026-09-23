"""Figures for the three-case alphabet sweep (notebook 241).

Data comes from 241_alphabet_ranking_collect.py: arms.csv and ranks.csv under
/Users/olga/data/botryllus/alphabet-ranking-three-cases/.

Every figure is a dot matrix: one row per alphabet (grouped by class count), seed
information in bits along x, one dot per arm. Colour is one sequential purple ramp for
one quantity per figure (rank, or E-value); a grey cross means the partner was not
among the hits; an open circle means the arm has no E-value because no Karlin-Altschul
fit was possible. The legend sits above the panels.
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
    "hp_lehninger2": 2, "hp_thomas_dill2": 2, "hp_kyte_doolittle2": 2,
    "hp_thomas_dill_no_c2": 2, "hp_lehninger_c_nonpolar2": 2, "hp_pbotc_1st_ed2": 2,
    "hp_lehninger_hpc3": 3, "gbmr4": 4, "polarity4": 4, "wwmj5": 5, "dayhoff6": 6,
    "gbmr7": 7, "funcgroups8": 8, "sdm12": 12, "mmseqs12": 12, "wass14": 14,
    "hsdm17": 17, "uniprot18": 18, "protein20": 20,
}
ORDER = sorted(CLASSES, key=lambda a: (CLASSES[a], a))
ROW = {a: i for i, a in enumerate(ORDER)}

METRICS5 = ["E-value", "mean IDF", "tf-idf", "enrichment", "Poisson p-value"]

# One hue for magnitude. Truncated so the light end is still visible on white.
# Light end truncated so it still shows on white. RAMP: more = darker. RANK_RAMP:
# rank 1 = darkest, so the best result is the mark the eye lands on.
RAMP = mpl.colors.LinearSegmentedColormap.from_list(
    "count", [plt.get_cmap("Purples")(x) for x in np.linspace(0.35, 1.0, 256)]
)
RANK_RAMP = RAMP.reversed()
NOT_FOUND = "#9a9a9a"
NO_EVALUE = "#6a3d9a"


def load() -> tuple[pl.DataFrame, pl.DataFrame]:
    # The first hundred rows of ranks.csv can be all-null in `rank`, so the types are
    # given rather than inferred.
    arms = pl.read_csv(DATA / "arms.csv", infer_schema_length=None)
    ranks = pl.read_csv(DATA / "ranks.csv", schema_overrides={
        "rank": pl.Int64, "n_tied": pl.Int64, "partner_value": pl.Float64,
        "best_value": pl.Float64, "partner_found": pl.Boolean, "top_gene": pl.Utf8})
    return arms, ranks


def _figure_with_legend_row(ncols: int, figsize: tuple, handles: list, sharey: bool = True,
                            width_ratios=None):
    """A figure whose top row is an axes holding only the legend, so tight_layout keeps
    the legend clear of the panels and the legend reads before the marks."""
    fig = plt.figure(figsize=figsize)
    n_lines = len(handles)
    gs = fig.add_gridspec(2, ncols, height_ratios=[0.05 * n_lines + 0.05, 1.0],
                          width_ratios=width_ratios, hspace=0.12)
    lax = fig.add_subplot(gs[0, :])
    lax.axis("off")
    lax.legend(handles=handles, loc="upper left", fontsize=9, frameon=False,
               borderaxespad=0.0, handletextpad=0.6)
    axes = []
    for i in range(ncols):
        ax = fig.add_subplot(gs[1, i], sharey=axes[0] if (axes and sharey) else None)
        if axes and sharey:
            plt.setp(ax.get_yticklabels(), visible=False)
        axes.append(ax)
    return fig, axes


def _rows_axis(ax, xlabel: bool = True):
    ax.set_yticks(range(len(ORDER)))
    ax.set_yticklabels([f"{a} ({CLASSES[a]})" for a in ORDER], fontsize=8)
    ax.set_ylim(len(ORDER) - 0.5, -0.5)
    ax.set_xlim(13, 47)
    ax.set_xticks([16, 20, 24, 28, 32, 36, 40, 44])
    if xlabel:
        ax.set_xlabel("information in one seed (bits = k × bits per position)", fontsize=9)
    ax.grid(axis="x", alpha=0.25)
    for y in (5.5, 6.5, 8.5, 9.5, 10.5, 11.5, 12.5, 14.5, 15.5, 16.5, 17.5):
        ax.axhline(y, color="#e5e5e5", lw=0.6, zorder=0)
    ax.tick_params(axis="x", labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)


def rank_matrix(ranks: pl.DataFrame, arms: pl.DataFrame, query: str, partner: str,
                path: Path, title_first_line: str, hypothesis: str, conclusion: str,
                annotate_le: int = 10) -> pl.DataFrame:
    """Five panels, one per metric: where `partner` ranks among human targets when
    `query` is searched, per alphabet (rows) and seed information (x)."""
    sub = ranks.filter((pl.col("query") == query) & pl.col("metric").is_in(METRICS5))
    fit = arms.select("alphabet", "ksize", "fitted", "searched")
    sub = sub.join(fit, on=["alphabet", "ksize"], how="left")
    norm = mpl.colors.LogNorm(vmin=1, vmax=10_000)

    handles = [
        Line2D([], [], marker="o", ls="", ms=8, color=RANK_RAMP(0.15), label=f"{partner} is among the hits; colour = its rank, number written when rank ≤ {annotate_le}"),
        Line2D([], [], marker="x", ls="", ms=7, color=NOT_FOUND, label=f"{partner} is not among the hits at this arm"),
        Line2D([], [], marker="o", ls="", ms=8, markerfacecolor="none", markeredgecolor=NO_EVALUE, label="E-value panel only: no E-value at this arm (no Karlin-Altschul fit possible)"),
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
                ax.scatter(x, y, s=46, facecolors="none", edgecolors=NO_EVALUE, lw=1.2, zorder=3)
                continue
            if not row["partner_found"]:
                ax.scatter(x, y, s=30, marker="x", color=NOT_FOUND, lw=1.0, zorder=3)
                continue
            r = row["rank"]
            ax.scatter(x, y, s=62, c=[RANK_RAMP(norm(r))], edgecolors="white", lw=0.4, zorder=4)
            if r <= annotate_le:
                ax.text(x, y, str(r), ha="center", va="center", fontsize=6.5,
                        color="white", zorder=5, fontweight="bold")
        ax.set_title(metric, fontsize=11)
        _rows_axis(ax, xlabel=(metric == METRICS5[2]))

    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=RANK_RAMP), ax=axes,
                      fraction=0.012, pad=0.01, aspect=30)
    cb.set_label(f"rank of {partner} among the human proteins hit (1 = best)", fontsize=9)
    cb.set_ticks([1, 10, 100, 1000, 10_000])
    cb.set_ticklabels(["1", "10", "100", "1_000", "10_000"])

    finish_figure(fig, path, tools=TOOLS, hypothesis=hypothesis, conclusion=conclusion,
                  title=title_first_line, header_y=1.01, footer_y=-0.02)
    return sub


def best_per_alphabet(ranks: pl.DataFrame, query: str) -> pl.DataFrame:
    """For each alphabet, the best rank the partner reaches over every k and every one
    of the five metrics, and which arm and metric got there."""
    sub = (ranks.filter((pl.col("query") == query) & pl.col("metric").is_in(METRICS5)
                        & pl.col("partner_found") & pl.col("rank").is_not_null())
           .sort("rank", "n_tied", "bits"))
    best = sub.group_by("alphabet").first().select(
        "alphabet", "rank", "n_tied", "n_targets", "ksize", "bits", "metric", "partner_value")
    found_any = sub.group_by("alphabet").agg(
        pl.col("bits").min().alias("found_from_bits"), pl.col("bits").max().alias("found_to_bits"))
    out = pl.DataFrame({"alphabet": ORDER, "classes": [CLASSES[a] for a in ORDER]})
    return out.join(best, on="alphabet", how="left").join(found_any, on="alphabet", how="left")


def best_rank_figure(ranks: pl.DataFrame, path: Path, hypothesis: str, conclusion: str) -> pl.DataFrame:
    """Two panels: the best rank BCL2 (gold standard) and CD47 (the reach) ever reach,
    per alphabet, over every k and metric."""
    handles = [
        Line2D([], [], marker="s", ls="", ms=9, color=RANK_RAMP(0.5), label="bar: best rank reached, over every k and every one of the five metrics; label says which"),
        Line2D([], [], marker="x", ls="", ms=7, color=NOT_FOUND, label="the partner is never among the hits for this alphabet at any k"),
    ]
    fig, axes = _figure_with_legend_row(2, (13, 7.0), handles)
    tables = []
    for ax, (q, partner, label) in zip(axes, [("Ced9", "BCL2", "gold standard: Ced9 query, BCL2 is the known homolog"),
                                             ("P66", "CD47", "the reach: P66 query, CD47 is the proposed partner")]):
        b = best_per_alphabet(ranks, q)
        tables.append(b.with_columns(pl.lit(q).alias("query")))
        for row in b.iter_rows(named=True):
            y = ROW[row["alphabet"]]
            if row["rank"] is None:
                ax.scatter(1.0, y, marker="x", color=NOT_FOUND, s=40, zorder=3)
                ax.text(1.25, y, "never among the hits", va="center", fontsize=7, color=NOT_FOUND)
                continue
            r = row["rank"]
            ax.barh(y, r, left=1, height=0.55, color=RANK_RAMP(0.5), zorder=2)
            ax.text(r * 1.12, y, f"{r:_} of {row['n_targets']:_}  ({row['metric']}, k={row['ksize']}"
                    + (f", {row['n_tied']} tied)" if row["n_tied"] else ")"),
                    va="center", fontsize=6.8)
        ax.set_xscale("log")
        ax.set_xlim(1, 2e7)
        ax.set_xticks([1, 10, 100, 1000, 10_000])
        ax.set_xticklabels(["1", "10", "100", "1_000", "10_000"])
        ax.set_xlabel(f"best rank of {partner} among the human proteins hit (1 = best)", fontsize=9)
        ax.set_title(label, fontsize=10.5)
        ax.set_yticks(range(len(ORDER)))
        ax.set_yticklabels([f"{a} ({CLASSES[a]})" for a in ORDER], fontsize=8)
        ax.set_ylim(len(ORDER) - 0.5, -0.5)
        ax.grid(axis="x", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    finish_figure(fig, path, tools=TOOLS, hypothesis=hypothesis, conclusion=conclusion,
                  header_y=1.01, footer_y=-0.02)
    return pl.concat(tables)


def bhf_matrix(ranks: pl.DataFrame, arms: pl.DataFrame, path: Path, hypothesis: str, conclusion: str) -> pl.DataFrame:
    """Two panels for BHF, which has no known partner: the best E-value any human
    protein reaches, and how many human proteins have any region at all."""
    e = (ranks.filter((pl.col("query") == "BHF") & (pl.col("metric") == "E-value"))
         .join(arms.select("alphabet", "ksize", "fitted", "searched", "n_targets_BHF"), on=["alphabet", "ksize"], how="left"))
    norm_e = mpl.colors.LogNorm(vmin=1e-3, vmax=1e3)
    ramp_e = RANK_RAMP  # dark = small E-value = more surprising
    handles = [
        Line2D([], [], marker="o", ls="", ms=8, color=RAMP(0.75), label="filled dot: colour = the panel's quantity"),
        Line2D([], [], marker="o", ls="", ms=8, markerfacecolor=ramp_e(0.1), markeredgecolor="black", markeredgewidth=1.2, label="black ring, left panel only: best E-value below 1, gene and E written beside it"),
        Line2D([], [], marker="x", ls="", ms=7, color=NOT_FOUND, label="no human protein has a region at this arm"),
        Line2D([], [], marker="o", ls="", ms=8, markerfacecolor="none", markeredgecolor=NO_EVALUE, label="left panel only: no E-value at this arm (no Karlin-Altschul fit possible)"),
    ]
    fig, axes = _figure_with_legend_row(2, (14, 8.0), handles)
    ax = axes[0]
    below_one: dict[int, list[str]] = {}
    for row in e.iter_rows(named=True):
        y, x = ROW[row["alphabet"]], row["bits"]
        if not row["searched"]:
            continue
        if row["fitted"] is not True:
            ax.scatter(x, y, s=46, facecolors="none", edgecolors=NO_EVALUE, lw=1.2, zorder=3)
            continue
        if row["best_value"] is None:
            ax.scatter(x, y, s=30, marker="x", color=NOT_FOUND, zorder=3)
            continue
        v = row["best_value"]
        ax.scatter(x, y, s=62, c=[ramp_e(norm_e(v))], edgecolors="black" if v < 1 else "white",
                   lw=1.2 if v < 1 else 0.4, zorder=4)
        if v < 1:
            below_one.setdefault(y, []).append(f"{row['top_gene']} E={v:.2g} at {x:.0f} bits")
    # Names go in the empty space right of the last arm, with a leader from the row.
    for y, labels in below_one.items():
        ax.text(51.5, y, "; ".join(labels), va="center", ha="right", fontsize=6.5, zorder=5,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.85))
    ax.set_title("best E-value of any human protein", fontsize=11)
    _rows_axis(ax)
    ax.set_xlim(13, 52)
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm_e, cmap=ramp_e), ax=ax, fraction=0.04, pad=0.02)
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
            ax.scatter(x, y, s=62, c=[RAMP(norm_n(n))], edgecolors="white", lw=0.4, zorder=4)
    ax.set_title("human proteins with at least one matched region", fontsize=11)
    _rows_axis(ax)
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm_n, cmap=RAMP), ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("number of human proteins hit (of 19_732)", fontsize=8)

    finish_figure(fig, path, tools=TOOLS, hypothesis=hypothesis, conclusion=conclusion,
                  header_y=1.01, footer_y=-0.02)
    return e.select("alphabet", "ksize", "bits", "fitted", "n_targets_BHF", "best_value", "top_gene").sort("alphabet", "ksize")


def pair_matrix(arms: pl.DataFrame, path: Path, hypothesis: str, conclusion: str) -> pl.DataFrame:
    """No database at all: how many exact k-mers the two proteins of each pair share,
    per alphabet and seed information."""
    norm = mpl.colors.LogNorm(vmin=1, vmax=100)
    handles = [
        Line2D([], [], marker="o", ls="", ms=8, color=RAMP(0.75), label="the two proteins share at least one exact k-mer; colour = how many"),
        Line2D([], [], marker="o", ls="", ms=8, markerfacecolor=RAMP(0.75), markeredgecolor="black", markeredgewidth=1.2, label="Ced9 panel only: a shared region lies in the BH3-binding groove (Ced9 150-195, BCL2 125-170)"),
        Line2D([], [], marker="x", ls="", ms=7, color=NOT_FOUND, label="no shared k-mer at this arm"),
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
            ax.scatter(x, y, s=62, c=[RAMP(norm(n))],
                       edgecolors="black" if in_win else "white", lw=1.2 if in_win else 0.4, zorder=4)
            if n <= 9:
                ax.text(x, y, str(n), ha="center", va="center", fontsize=6.5,
                        color="white" if n >= 4 else "#1a1a1a", zorder=5, fontweight="bold")
        ax.set_title(f"{q} vs {partner}: exact k-mers the two proteins share", fontsize=10.5)
        _rows_axis(ax)
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=RAMP), ax=axes, fraction=0.02, pad=0.02)
    cb.set_label("shared k-mers (number written when ≤ 9)", fontsize=8)
    finish_figure(fig, path, tools=TOOLS.replace("index + search + pair", "pair (no database)"),
                  hypothesis=hypothesis, conclusion=conclusion, header_y=1.01, footer_y=-0.02)
    return arms.select("alphabet", "ksize", "bits", "pair_Ced9_shared_kmers", "pair_Ced9_in_window",
                       "pair_P66_shared_kmers").sort("alphabet", "ksize")


def partner_evalue_figure(ranks: pl.DataFrame, arms: pl.DataFrame, path: Path,
                          hypothesis: str, conclusion: str) -> pl.DataFrame:
    """Two panels in the row layout: the E-value BCL2 (left) and CD47 (right) themselves
    get at every arm where they are among the hits and an E-value exists. The best one
    per alphabet is written at the right edge."""
    norm = mpl.colors.LogNorm(vmin=1e-1, vmax=1e6)
    handles = [
        Line2D([], [], marker="o", ls="", ms=8, color=RANK_RAMP(0.5), label="the partner is among the hits and has an E-value; colour = that E-value (dark = below 1)"),
        Line2D([], [], marker="x", ls="", ms=7, color=NOT_FOUND, label="the partner is not among the hits at this arm, or the arm has no E-value"),
    ]
    fig, axes = _figure_with_legend_row(2, (13.5, 8.0), handles)
    tables = []
    for ax, (q, partner) in zip(axes, [("Ced9", "BCL2"), ("P66", "CD47")]):
        sub = (ranks.filter((pl.col("query") == q) & (pl.col("metric") == "E-value"))
               .join(arms.select("alphabet", "ksize", "fitted", "searched"), on=["alphabet", "ksize"], how="left")
               .filter(pl.col("searched")))
        best: dict[int, float] = {}
        for row in sub.iter_rows(named=True):
            y, x = ROW[row["alphabet"]], row["bits"]
            ok = row["fitted"] is True and row["partner_found"] and row["partner_value"] is not None
            if not ok:
                ax.scatter(x, y, s=30, marker="x", color=NOT_FOUND, zorder=3)
                continue
            v = row["partner_value"]
            ax.scatter(x, y, s=62, c=[RANK_RAMP(norm(v))], edgecolors="white", lw=0.4, zorder=4)
            best[y] = min(best.get(y, float("inf")), v)
        for y, v in best.items():
            ax.text(51.5, y, f"best E = {v:_.0f}" if v >= 10 else f"best E = {v:.2g}",
                    va="center", ha="right", fontsize=7.5)
        ax.set_title(f"{partner}'s own E-value when {q} is the query", fontsize=10.5)
        _rows_axis(ax)
        ax.set_xlim(13, 52)
        tables.append(sub.filter(pl.col("partner_found") & (pl.col("fitted") == True))  # noqa: E712
                      .select("query", "alphabet", "ksize", "bits", "partner_value", "rank", "n_targets"))
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=RANK_RAMP), ax=axes, fraction=0.02, pad=0.02)
    cb.set_label("E-value of the partner's best region (lower is more surprising; 1 = chance)", fontsize=8)
    finish_figure(fig, path, tools=TOOLS, hypothesis=hypothesis, conclusion=conclusion,
                  header_y=1.01, footer_y=-0.02)
    return pl.concat(tables)


def lambda_zero_figure(arms: pl.DataFrame, path: Path, hypothesis: str, conclusion: str) -> pl.DataFrame:
    """Which regions can get an E-value at all. Left: the share of regions at each arm
    whose per-region lambda is 0 (E = inf) because the region's own identity is above
    C / (1 + C). Right, one bar per alphabet: the expected score of a chance position,
    p - C (1 - p), with the database's chance match probability p and the alphabet's
    penalty C; at or above 0 no lambda exists for any region."""
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    handles = [
        Line2D([], [], marker="o", ls="", ms=8, color=RAMP(0.75), label="left: colour = share of the arm's regions with no E-value (per-region lambda 0)"),
        Line2D([], [], marker="x", ls="", ms=7, color=NOT_FOUND, label="left: no region at this arm for any of the three queries"),
        Line2D([], [], marker="s", ls="", ms=9, color=RAMP(0.75), label="right: expected score of one chance position at this alphabet's penalty; at or above the zero line no lambda can exist"),
    ]
    fig, axes = _figure_with_legend_row(2, (14, 8.0), handles, sharey=True, width_ratios=[1.6, 1.0])
    ax = axes[0]
    for row in arms.iter_rows(named=True):
        y, x = ROW[row["alphabet"]], row["bits"]
        f = row.get("frac_lambda_zero")
        if f is None:
            ax.scatter(x, y, s=30, marker="x", color=NOT_FOUND, zorder=3)
        else:
            ax.scatter(x, y, s=62, c=[RAMP(norm(f))], edgecolors="white", lw=0.4, zorder=4)
    ax.set_title("share of regions with no E-value, by arm", fontsize=11)
    _rows_axis(ax)
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=RAMP), ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("share of regions with lambda 0 (E = inf)", fontsize=8)

    ax = axes[1]
    per = (arms.group_by("alphabet").agg(pl.col("chance_drift").first(), pl.col("p_match").first(),
                                          pl.col("penalty").first()))
    for row in per.iter_rows(named=True):
        y = ROW[row["alphabet"]]
        d = row["chance_drift"]
        if d is None:
            continue
        ax.barh(y, d, height=0.55, color=RAMP(0.75), zorder=2)
        # Labels sit right of the zero line (or of a positive bar), never on a bar.
        ax.text(max(d, 0) + 0.03, y, f"p = {row['p_match']:.2f}, C = {row['penalty']:g}",
                va="center", ha="left", fontsize=7)
    ax.axvline(0, color="#c0392b", ls="--", lw=1.2)
    ax.set_xlim(-1.6, 0.9)
    ax.set_xlabel("expected score of a chance position: p − C (1 − p)", fontsize=9)
    ax.set_title("can a lambda exist at all?", fontsize=11)
    ax.grid(axis="x", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    finish_figure(fig, path, tools=TOOLS, hypothesis=hypothesis, conclusion=conclusion,
                  header_y=1.01, footer_y=-0.02)
    return per.sort("chance_drift", descending=True)
