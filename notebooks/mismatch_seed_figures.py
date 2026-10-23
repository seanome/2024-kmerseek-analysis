"""Loading and figures for notebook 246 (seeds that allow mismatches).

Reads only the tables written by scripts/run_246_mismatch_seeds.py, so the notebook runs
in the 2025-kmerseek-analysis env without numba.

One colour per seed family in every figure:
  exact   blue            k class matches in a row (kmerseek today)
  spaced  orange          class matches at the 1s of a pattern; the 0s may mismatch
  chained bluish green    n exact k-mers on one diagonal inside W positions
  window  reddish purple  at least m class matches among L positions on one diagonal
Vermillion is what a search of the human proteome needs (1 or 10 chance seeds per search).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.lines import Line2D

import pubfig as pf

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
FIG = ROOT / "figures"

FAMILY_COLOUR = {
    "exact": pf.OKABE_ITO["blue"],
    "spaced": pf.OKABE_ITO["orange"],
    "chained": pf.OKABE_ITO["bluish_green"],
    "window": pf.OKABE_ITO["reddish_purple"],
}
FAMILY_LABEL = {
    "exact": "exact: k class matches in a row",
    "spaced": "spaced: matches at the 1s of a pattern",
    "chained": "chained: n exact k-mers within W positions",
    "window": "window: at least m matches among L positions",
}
SCALED_MARKER = {1: "o", 2: "^"}
NEED = pf.OKABE_ITO["vermillion"]
NO_SEED = pf.GREY
ZERO_X = 0.3  # where a scheme with no chance seed at all is drawn on a ratio axis

ALPHABET_CLASSES = {
    "hp_thomas_dill2": "H = ACFILMVWY, P = the rest",
    "hp_lehninger2": "H = AFGILMPVWY, P = the rest",
}
ALPHABETS = list(ALPHABET_CLASSES)
QUERIES = {"CED-9": "BCL2", "P66": "CD47"}


def read(name: str) -> pl.DataFrame:
    return pl.read_csv(TABLES / name, separator="\t")


def schemes() -> pl.DataFrame:
    return read("246_seed_schemes.tsv")


def search() -> pl.DataFrame:
    return read("246_human_proteome_search.tsv").join(
        schemes().select("scheme", "family", "scaled", "weight", "span"), on="scheme"
    )


def reach(identity_bin: str = "20-30%") -> pl.DataFrame:
    return read("246_pfam_seed_pairs_reach.tsv").filter(
        pl.col("identity_bin") == identity_bin
    )


def frontier_table(query: str = "CED-9") -> pl.DataFrame:
    """Per alphabet and keyed or window scheme: chance seeds per search and Pfam reach."""
    return (
        search()
        .filter((pl.col("query") == query) & (pl.col("family") != "extend"))
        .join(
            reach().select(
                "alphabet", "scheme", "n_pairs", "fraction_of_pairs_with_seed"
            ),
            on=["alphabet", "scheme"],
        )
        .select(
            "alphabet",
            "family",
            "scheme",
            "scaled",
            "weight",
            "span",
            "chance_seed_positions_per_search",
            "n_proteins_with_seed",
            "fraction_of_pairs_with_seed",
            "n_pairs",
        )
        .sort("alphabet", "chance_seed_positions_per_search")
    )


def plain_count(v: float) -> str:
    """1, 10, 100, 1,000, 10,000, 100,000, 1 M, 10 M, 100 M."""
    return f"{v / 1e6:,.0f} M" if v >= 1e6 else f"{v:,.0f}"


def _ratio_axis_x(ax, xmax):
    ax.set_xscale("log")
    ticks = [ZERO_X] + [10.0**e for e in range(0, int(np.ceil(np.log10(xmax))) + 1)]
    ax.set_xticks(ticks, ["0"] + [plain_count(t) for t in ticks[1:]], rotation=0)
    ax.minorticks_off()
    ax.set_xlim(ZERO_X / 1.6, xmax * 1.6)


def _family_handles(families, with_scaled=True):
    h = [
        Line2D([], [], ls="", marker="o", color=FAMILY_COLOUR[f], label=FAMILY_LABEL[f])
        for f in families
    ]
    if with_scaled:
        h += [
            Line2D(
                [],
                [],
                ls="",
                marker="o",
                color="black",
                label="scaled 1 (every seed kept)",
            ),
            Line2D(
                [],
                [],
                ls="",
                marker="^",
                color="black",
                label="scaled 2 (half the seed keys kept)",
            ),
        ]
    return h


def _need_lines(ax, label=True):
    ax.axvline(1, color=NEED, lw=1, label="1 chance seed per search" if label else None)
    ax.axvline(
        10,
        color=NEED,
        lw=1,
        ls="--",
        label="10 chance seeds per search" if label else None,
    )


def fig_frontier(t: pl.DataFrame, path: Path) -> None:
    """Reach on Pfam 20-30% pairs against chance seeds per search, one point per scheme."""
    fig, axs = pf.figure(pf.TWO_COLUMN_MM, 80, ncols=2, sharey=True)
    xmax = t["chance_seed_positions_per_search"].max()
    for ax, alphabet, letter in zip(axs, ALPHABETS, "ab"):
        d = t.filter(pl.col("alphabet") == alphabet)
        _need_lines(ax, label=letter == "a")
        for r in d.iter_rows(named=True):
            x = max(r["chance_seed_positions_per_search"], ZERO_X)
            ax.plot(
                x,
                100 * r["fraction_of_pairs_with_seed"],
                ls="",
                marker=SCALED_MARKER[r["scaled"]],
                ms=3.5,
                color=FAMILY_COLOUR[r["family"]],
                alpha=0.9,
            )
        ex = d.filter((pl.col("family") == "exact") & (pl.col("scaled") == 1)).sort(
            "weight"
        )
        ax.plot(
            np.maximum(ex["chance_seed_positions_per_search"], ZERO_X),
            100 * ex["fraction_of_pairs_with_seed"],
            color=FAMILY_COLOUR["exact"],
            lw=0.6,
        )
        for r in ex.filter(pl.col("weight").is_in([12, 20, 28])).iter_rows(named=True):
            ax.annotate(
                f"k = {r['weight']}",
                (
                    max(r["chance_seed_positions_per_search"], ZERO_X),
                    100 * r["fraction_of_pairs_with_seed"],
                ),
                xytext=(-6, 8),
                textcoords="offset points",
                ha="right",
                arrowprops={"arrowstyle": "-", "lw": 0.4, "color": "black"},
            )
        _ratio_axis_x(ax, xmax)
        ax.set_xlabel(
            "chance seeds per search: seed positions on the 19,731 other\n"
            "human proteins, CED-9 (280 aa) as the query"
        )
        ax.set_title(
            f"seed scan of this notebook (not kmerseek), {alphabet}\n({ALPHABET_CLASSES[alphabet]})",
            loc="left",
        )
        pf.panel_label(ax, letter)
    axs[0].set_ylabel(
        "Pfam seed pairs at 20-30% identity\nwith a seed on their alignment (%)"
    )
    axs[0].set_ylim(0, 100)
    handles = _family_handles(FAMILY_COLOUR) + axs[0].get_legend_handles_labels()[0]
    fig.legend(handles=handles, loc="outside upper center", ncol=4, frameon=False)
    pf.save(fig, path, formats=("pdf", "png"))


def fig_partner_rank(s: pl.DataFrame, path: Path) -> None:
    """Partner rank against chance seeds per search; one panel per query and alphabet."""
    fig, axs = pf.figure(pf.TWO_COLUMN_MM, 120, nrows=2, ncols=2, sharey=True)
    xmax = s["chance_seed_positions_per_search"].max()
    n = int(s["n_proteins"].max())
    for row, (query, partner) in enumerate(QUERIES.items()):
        for col, alphabet in enumerate(ALPHABETS):
            ax = axs[row, col]
            d = s.filter((pl.col("query") == query) & (pl.col("alphabet") == alphabet))
            ax.axhspan(
                0.7,
                10,
                color=pf.OKABE_ITO["yellow"],
                alpha=0.35,
                lw=0,
                label="top 10 of 19,732 human proteins",
            )
            _need_lines(ax, label=(row, col) == (0, 0))
            ext = d.filter(pl.col("family") == "extend").row(0, named=True)
            ax.axhline(
                ext["partner_rank"],
                color="black",
                ls="-.",
                lw=0.8,
                label="best ungapped segment, no seed (+1 match, -2 mismatch)",
            )
            for r in d.filter(pl.col("family") != "extend").iter_rows(named=True):
                x = max(r["chance_seed_positions_per_search"], ZERO_X)
                if r["partner_rank"] is None:
                    ax.plot(x, n * 1.6, ls="", marker="x", ms=3, color=NO_SEED)
                else:
                    ax.plot(
                        x,
                        r["partner_rank"],
                        ls="",
                        marker=SCALED_MARKER[r["scaled"]],
                        ms=3.5,
                        color=FAMILY_COLOUR[r["family"]],
                        alpha=0.9,
                    )
            _ratio_axis_x(ax, xmax)
            ax.set_yscale("log")
            ax.set_yticks(
                [1, 10, 100, 1_000, 10_000, n * 1.6],
                ["1", "10", "100", "1,000", "10,000", "no seed\non partner"],
            )
            ax.minorticks_off()
            ax.set_ylim(n * 3, 0.7)
            ax.set_title(
                f"{query} query against the human proteome: rank of {partner}; "
                f"{alphabet}, seed scan of this notebook",
                loc="left",
            )
            if row == 1:
                ax.set_xlabel(
                    f"chance seeds per search (seed positions on the other human proteins)"
                )
            if col == 0:
                ax.set_ylabel(f"rank of the partner among\n19,732 human proteins")
            pf.panel_label(ax, "abcd"[2 * row + col])
    handles = (
        _family_handles(FAMILY_COLOUR)
        + [
            Line2D(
                [], [], ls="", marker="x", color=NO_SEED, label="partner has no seed"
            )
        ]
        + axs[0, 0].get_legend_handles_labels()[0]
    )
    fig.legend(handles=handles, loc="outside upper center", ncol=4, frameon=False)
    pf.save(fig, path, formats=("pdf", "png"))


def fig_random_null(draws: pl.DataFrame, null: pl.DataFrame, path: Path) -> None:
    """Best rank over all 59 schemes: the partner against a protein drawn at random."""
    fig, axs = pf.figure(pf.TWO_COLUMN_MM, 50, ncols=4, sharey=True)
    bins = np.logspace(0, np.log10(20_000), 40)
    i = 0
    for query, partner in QUERIES.items():
        for alphabet in ALPHABETS:
            ax = axs[i]
            d = draws.filter(
                (pl.col("query") == query) & (pl.col("alphabet") == alphabet)
            )
            r = null.filter(
                (pl.col("query") == query) & (pl.col("alphabet") == alphabet)
            ).row(0, named=True)
            ax.hist(
                d["random_protein_best_rank"],
                weights=d["n_draws"] / d["n_draws"].sum() * 100,
                bins=bins,
                color=pf.GREY,
                label="a human protein drawn at random (20,000 draws)",
            )
            ax.axvline(
                r["partner_best_rank_any_scheme"],
                color="black",
                lw=1.2,
                label="the partner (BCL2 or CD47)",
            )
            ax.set_xscale("log")
            ax.set_xticks(
                [1, 10, 100, 1_000, 10_000], ["1", "10", "100", "1,000", "10,000"]
            )
            ax.minorticks_off()
            ax.set_title(
                f"{query}: {partner}, {alphabet}\nrank {r['partner_best_rank_any_scheme']:,.0f}; "
                f"random as good: {100 * r['p_random_protein_at_least_as_good']:.0f}%",
                loc="left",
            )
            ax.set_xlabel("best rank over 59 seed schemes")
            pf.panel_label(ax, "abcd"[i])
            i += 1
    axs[0].set_ylabel("draws (%)")
    pf.shared_legend(fig, frameon=False)
    pf.save(fig, path, formats=("pdf", "png"))


def scaled_pairs() -> pl.DataFrame:
    """Each keyed scheme at scaled 1 next to the same seed shape at scaled 2."""
    idx = read("246_human_proteome_index_entries.tsv")
    r = reach().select("alphabet", "scheme", "fraction_of_pairs_with_seed")
    s = (
        search()
        .filter(pl.col("query") == "CED-9")
        .select(
            "alphabet", "scheme", "family", "scaled", "chance_seed_positions_per_search"
        )
    )
    t = s.join(
        idx.select("alphabet", "scheme", "index_entries"), on=["alphabet", "scheme"]
    ).join(r, on=["alphabet", "scheme"])
    t = t.with_columns(pl.col("scheme").str.replace(", scaled 2", "").alias("shape"))
    one = t.filter(pl.col("scaled") == 1).drop("scaled", "scheme")
    two = t.filter(pl.col("scaled") == 2).drop("scaled", "scheme", "family")
    return (
        one.join(two, on=["alphabet", "shape"], suffix="_scaled2")
        .with_columns(
            (pl.col("index_entries_scaled2") / pl.col("index_entries")).alias(
                "index_ratio"
            ),
            (
                pl.col("fraction_of_pairs_with_seed_scaled2")
                / pl.col("fraction_of_pairs_with_seed")
            ).alias("reach_ratio"),
        )
        .sort("alphabet", "family", "shape")
    )


def fig_scaled(t: pl.DataFrame, path: Path) -> None:
    fig, axs = pf.figure(pf.TWO_COLUMN_MM, 70, ncols=2)
    a, b = axs
    for r in t.iter_rows(named=True):
        c = FAMILY_COLOUR[r["family"]]
        mk = "o" if r["alphabet"] == "hp_thomas_dill2" else "s"
        a.plot(
            r["index_entries"] / 1e6,
            r["index_entries_scaled2"] / 1e6,
            ls="",
            marker=mk,
            ms=3.5,
            color=c,
            alpha=0.85,
        )
        b.plot(
            100 * r["fraction_of_pairs_with_seed"],
            100 * r["fraction_of_pairs_with_seed_scaled2"],
            ls="",
            marker=mk,
            ms=3.5,
            color=c,
            alpha=0.85,
        )
    a.plot(
        [0, 12],
        [0, 6],
        color="black",
        lw=0.6,
        ls="--",
        label="half (scaled 2 = scaled 1 / 2)",
    )
    a.set_xlim(10, 11.6)
    a.set_ylim(4.5, 7)
    a.set_xlabel("index entries on the human proteome, scaled 1 (millions)")
    a.set_ylabel("index entries, scaled 2 (millions)")
    b.plot([0, 100], [0, 100], color="black", lw=0.6, ls=":", label="no change (y = x)")
    b.set_xlim(0, 100)
    b.set_ylim(0, 100)
    b.set_xlabel("Pfam 20-30% pairs with a seed, scaled 1 (%)")
    b.set_ylabel("Pfam 20-30% pairs with a seed, scaled 2 (%)")
    for ax, letter in zip(axs, "ab"):
        pf.panel_label(ax, letter)
    handles = (
        _family_handles([f for f in FAMILY_COLOUR if f != "window"], with_scaled=False)
        + [
            Line2D([], [], ls="", marker="o", color="black", label="hp_thomas_dill2"),
            Line2D([], [], ls="", marker="s", color="black", label="hp_lehninger2"),
        ]
        + a.get_legend_handles_labels()[0]
        + b.get_legend_handles_labels()[0]
    )
    fig.legend(handles=handles, loc="outside upper center", ncol=4, frameon=False)
    pf.save(fig, path, formats=("pdf", "png"))


# ------------------------------------------------------------------ BH1 diagonal -------
BH1_SCHEMES = [
    "exact k=12",
    "exact k=18",
    "exact k=20",
    "spaced 11011011011011011011011",
    "spaced 111010010100110111",
    "chained 2 x k=8 in 40",
    "chained 3 x k=8 in 60",
    "window 18 of 20",
    "window 27 of 30",
    "window 36 of 40",
]
BH1_DIAGONAL = 130 - 154  # BCL2 position - CED-9 position (notebook 245)
SHOW_CED9 = (140, 200)  # CED-9 positions drawn, 1-based inclusive


def ced9_and_bcl2() -> tuple[str, str]:
    seqs, name = {}, None
    for line in (TABLES / "246_queries_from_241.fasta").read_text().splitlines():
        if line.startswith(">"):
            name = line[1:].split()[0]
            seqs[name] = ""
        else:
            seqs[name] += line.strip()
    bcl2 = "".join(
        (ROOT / "notebooks/test-data/bcl2.fasta").read_text().splitlines()[1:]
    )
    return seqs["Ced9"], bcl2


def hp_string(seq: str, alphabet: str) -> str:
    h = set(ALPHABET_CLASSES[alphabet].split(",")[0].split("=")[1].strip())
    return "".join("H" if c in h else "P" for c in seq)


def bh1_rows(alphabet: str) -> dict:
    ced9, bcl2 = ced9_and_bcl2()
    a, b = SHOW_CED9
    q = ced9[a - 1 : b]
    t = bcl2[a - 1 + BH1_DIAGONAL : b + BH1_DIAGONAL]
    qc, tc = hp_string(q, alphabet), hp_string(t, alphabet)
    return {
        "ced9": q,
        "bcl2": t,
        "ced9_classes": qc,
        "bcl2_classes": tc,
        "match": "".join("|" if x == y else " " for x, y in zip(qc, tc)),
        "identical": "".join("|" if x == y else " " for x, y in zip(q, t)),
    }


def print_bh1(alphabet: str) -> None:
    r = bh1_rows(alphabet)
    a, b = SHOW_CED9
    print(
        f"{alphabet} ({ALPHABET_CLASSES[alphabet]}); BH1 is CED-9 154-190 / BCL2 130-166"
    )
    print(f"CED-9    {a:>4} {r['ced9']} {b}")
    print(f"              {r['identical']}   identical residues")
    print(f"BCL2     {a + BH1_DIAGONAL:>4} {r['bcl2']} {b + BH1_DIAGONAL}")
    print(f"CED-9 HP {a:>4} {r['ced9_classes']}")
    print(
        f"              {r['match']}   same class: {r['match'].count('|')} of {len(r['match'])}"
    )
    print(f"BCL2 HP  {a + BH1_DIAGONAL:>4} {r['bcl2_classes']}")


def fig_bh1(pos: pl.DataFrame, s: pl.DataFrame, path: Path) -> None:
    """Residues of CED-9 and BCL2 on the BH1 diagonal, and where each seed fires."""
    pf.use_style()
    fig, axs = pf.figure(pf.TWO_COLUMN_MM, 150, nrows=2, layout=None)
    # Row labels are text in data coordinates, which constrained layout does not see, so
    # the left margin is set by hand to keep the figure 183 mm wide.
    fig.subplots_adjust(left=0.37, right=0.99, top=0.88, bottom=0.06, hspace=0.32)
    a, b = SHOW_CED9
    x = np.arange(a, b + 1)
    for ax, alphabet, letter in zip(axs, ALPHABETS, "ab"):
        r = bh1_rows(alphabet)
        seq_rows = [
            ("CED-9 residues", r["ced9"]),
            ("BCL2 residues", r["bcl2"]),
            ("CED-9 H/P class", r["ced9_classes"]),
            ("BCL2 H/P class", r["bcl2_classes"]),
        ]
        y = 0
        ax.axvspan(
            154 - 0.5, 190 + 0.5, color=pf.OKABE_ITO["sky_blue"], alpha=0.18, lw=0
        )
        for label, text in seq_rows:
            for xi, ch in zip(x, text):
                ax.text(
                    xi, y, ch, ha="center", va="center", family="monospace", fontsize=6
                )
            ax.text(a - 1.5, y, label, ha="right", va="center")
            y -= 1
            if label == "CED-9 H/P class":
                for xi, m in zip(x, r["match"]):
                    if m == "|":
                        ax.plot([xi, xi], [y + 0.3, y - 0.3], color="black", lw=0.6)
                ax.text(
                    a - 1.5,
                    y,
                    f"same class ({r['match'].count('|')} of {len(x)})",
                    ha="right",
                    va="center",
                )
                y -= 1
        y -= 0.6
        sub = pos.filter(pl.col("alphabet") == alphabet)
        chance = s.filter(
            (pl.col("query") == "CED-9") & (pl.col("alphabet") == alphabet)
        )
        for name in BH1_SCHEMES:
            fam = name.split()[0]
            ends = sub.filter(pl.col("scheme") == name)["ced9_position"].to_list()
            ends = [e for e in ends if a <= e <= b]
            c = chance.filter(pl.col("scheme") == name).row(0, named=True)
            ax.axhline(y, color="#dddddd", lw=0.3, zorder=0)
            ax.plot(
                ends, [y] * len(ends), ls="", marker="o", ms=3, color=FAMILY_COLOUR[fam]
            )
            ax.text(
                a - 1.5,
                y,
                f"{name}   ({c['chance_seed_positions_per_search']:,} chance seeds)",
                ha="right",
                va="center",
            )
            y -= 1
        ax.set_ylim(y + 0.3, 0.7)
        ax.set_xlim(a - 0.8, b + 0.8)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_xticks(np.arange(a, b + 1, 5))
        ax.set_xlabel(f"CED-9 position (BCL2 position = CED-9 - {-BH1_DIAGONAL})")
        ax.set_title(
            f"CED-9 against BCL2 on the BH1 diagonal; {alphabet} "
            f"({ALPHABET_CLASSES[alphabet]})",
            loc="left",
        )
        pf.panel_label(ax, letter, dx_pt=-140)
    handles = [
        Line2D(
            [],
            [],
            ls="",
            marker="o",
            color=FAMILY_COLOUR[f],
            label=(
                f"{FAMILY_LABEL[f]} (dot = last position of a seed)"
                if f == "exact"
                else FAMILY_LABEL[f]
            ),
        )
        for f in FAMILY_COLOUR
    ]
    handles.append(
        plt.Rectangle(
            (0, 0),
            1,
            1,
            color=pf.OKABE_ITO["sky_blue"],
            alpha=0.3,
            label="BH1 block, CED-9 154-190 / BCL2 130-166",
        )
    )
    handles.append(
        Line2D(
            [],
            [],
            ls="",
            label="(n chance seeds) = seed positions of that scheme "
            "on the other 19,731 human proteins, CED-9 query",
        )
    )
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=2,
        frameon=False,
    )
    pf.save(fig, path, formats=("pdf", "png"))
