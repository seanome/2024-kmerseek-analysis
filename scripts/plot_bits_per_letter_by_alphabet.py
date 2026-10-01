#!/usr/bin/env python3
"""Bits of information per letter for each of kmerseek's 19 alphabets, three ways.

Question: how much does one matching letter tell you, in each alphabet? A seed of k letters
carries k times that, so this number sets how long a seed each alphabet needs.

Three values per alphabet, all in bits per letter:
  log2(n)      n = number of letters. The value if every letter were equally common.
  composition  -log2(sum over letters of p^2), where p is the share of one letter in
               Swiss-Prot. sum p^2 is the chance that two residues drawn at random fall in the
               same letter, so this is the bits one matching letter is worth when residues
               are drawn by their real frequencies.
  human        measured from kmerseek's own seed counts in the human proteome. P(k) is the
               mean number of proteins sharing a seed of k letters (one seed drawn at random).
               Each extra letter divides P by 2^B, so
                   B = [log2(P(k_small) - 1) - log2(P(k_main) - 1)] / (k_main - k_small)
               (the -1 removes the protein the seed came from).

Inputs:
  tables/250_two_k_per_alphabet.csv   k_small, k_main and P at each, written by
                                      scripts/elm_seed_floor.py (notebook 250)
  notebooks/hp_conservation_utils.ALPHABET_CLUSTERS   letter groups for 15 alphabets
  ALPHABET_CLUSTERS_EXTRA below       the other 4, copied from kmerseek v0.4.0
                                      src/rust/alphabets.rs
  SWISSPROT_PERCENT below             Swiss-Prot release 2026_03 composition, section 6.1 of
                                      https://web.expasy.org/docs/relnotes/relstat.html

Outputs:
  figures/bits_per_letter_by_alphabet_swissprot_human.{png,pdf,svg}
  tables/bits_per_letter_by_alphabet.csv   the numbers the figure draws

Run: python scripts/plot_bits_per_letter_by_alphabet.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import polars as pl
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "notebooks"))
import hp_conservation_utils as hc  # noqa: E402

# The four alphabets hp_conservation_utils does not list, from kmerseek v0.4.0
# src/rust/alphabets.rs (FUNCGROUPS8_CLUSTERS, MMSEQS12_CLUSTERS, WASS14_CLUSTERS,
# HSDM17_CLUSTERS).
ALPHABET_CLUSTERS_EXTRA = {
    "funcgroups8": ["GVALI", "ST", "CM", "FY", "WHP", "NQ", "DE", "KR"],
    "mmseqs12": ["AST", "LM", "IV", "KR", "EQ", "ND", "FY", "C", "G", "H", "P", "W"],
    "wass14": [
        "WM",
        "DI",
        "P",
        "C",
        "AV",
        "K",
        "T",
        "RE",
        "G",
        "L",
        "Y",
        "SH",
        "F",
        "NQ",
    ],
    "hsdm17": [
        "A",
        "D",
        "KE",
        "R",
        "N",
        "T",
        "S",
        "Q",
        "Y",
        "F",
        "LIV",
        "M",
        "C",
        "W",
        "H",
        "G",
        "P",
    ],
}
CLUSTERS = {**hc.ALPHABET_CLUSTERS, **ALPHABET_CLUSTERS_EXTRA}

# UniProtKB/Swiss-Prot release 2026_03 (02-Sep-2026), 209_017_843 residues,
# "6.1 Composition in percent for the complete database".
SWISSPROT_RELEASE = "2026_03"
SWISSPROT_PERCENT = {
    "A": 8.25, "Q": 3.93, "L": 9.64, "S": 6.66,
    "R": 5.52, "E": 6.71, "K": 5.79, "T": 5.36,
    "N": 4.06, "G": 7.07, "M": 2.41, "W": 1.10,
    "D": 5.46, "H": 2.27, "F": 3.86, "Y": 2.92,
    "C": 1.38, "I": 5.90, "P": 4.75, "V": 6.85,
}  # fmt: skip

# Alphabet size groups, top to bottom, with a gap between groups.
SIZE_GROUPS = [
    ("20 letters", 20, 20),
    ("12–18 letters", 12, 18),
    ("4–8 letters", 4, 8),
    ("2–3 letters", 2, 3),
]

NOMINAL = "#8a909b"  # log2 n, open circle
COMPOSITION = "#2b55c7"  # Swiss-Prot composition, filled circle
HUMAN = "#c0601a"  # human proteome seed counts, diamond
JOIN = "#c3c8d0"
GRID = "#e6e8ec"
GROUP_INK = "#555b66"
MONO = "DejaVu Sans Mono"


def bits_from_composition(clusters: list[str]) -> float:
    total = sum(SWISSPROT_PERCENT.values())
    shares = [sum(SWISSPROT_PERCENT[aa] for aa in c) / total for c in clusters]
    return -math.log2(sum(p * p for p in shares))


def bits_from_human_seed_counts(row: dict) -> float:
    drop = math.log2(row["proteins_per_seed_k_small"] - 1) - math.log2(
        row["proteins_per_seed_k_main"] - 1
    )
    return drop / (row["k_main"] - row["k_small"])


def letter_groups_text(name: str, clusters: list[str]) -> str:
    if len(clusters) == 20:
        return "each amino acid its own letter"
    if len(clusters) >= 12:
        return " ".join(c for c in clusters if len(c) > 1) + "  (rest single)"
    if len(clusters) <= 3:
        return " ".join(clusters)  # HP tables list hydrophobic first, polar second
    return " ".join(sorted(clusters, key=len, reverse=True))


def build_table() -> pl.DataFrame:
    seeds = pl.read_csv(REPO / "tables" / "250_two_k_per_alphabet.csv")
    missing = set(seeds["alphabet"]) ^ set(CLUSTERS)
    if missing:
        raise ValueError(
            f"alphabets in only one of the seed table and the letter groups: {missing}"
        )
    rows = []
    for r in seeds.iter_rows(named=True):
        clusters = CLUSTERS[r["alphabet"]]
        n = len(clusters)
        rows.append(
            {
                "alphabet": r["alphabet"],
                "n_letters": n,
                "bits_log2_n": math.log2(n),
                "bits_swissprot_composition": bits_from_composition(clusters),
                "bits_human_seed_counts": bits_from_human_seed_counts(r),
                "k_small": r["k_small"],
                "k_main": r["k_main"],
                "proteins_per_seed_k_small": r["proteins_per_seed_k_small"],
                "proteins_per_seed_k_main": r["proteins_per_seed_k_main"],
                "letter_groups": " ".join(clusters),
            }
        )
    return pl.DataFrame(rows).sort(["n_letters", "alphabet"], descending=[True, False])


def draw_row(ax, yy: float, r: dict) -> None:
    vals = [
        r["bits_log2_n"],
        r["bits_swissprot_composition"],
        r["bits_human_seed_counts"],
    ]
    ax.axhline(yy, color=GRID, lw=0.6, zorder=0)
    ax.plot(
        [min(vals), max(vals)],
        [yy, yy],
        color=JOIN,
        lw=1.4,
        zorder=1,
        solid_capstyle="butt",
    )
    # The open circle is drawn larger so a filled circle at nearly the same value sits
    # inside it and both stay visible.
    ax.scatter(
        r["bits_log2_n"],
        yy,
        s=44,
        facecolor="white",
        edgecolor=NOMINAL,
        linewidth=1.1,
        zorder=2,
    )
    ax.scatter(
        r["bits_swissprot_composition"],
        yy,
        s=18,
        color=COMPOSITION,
        linewidth=0,
        zorder=3,
    )
    ax.scatter(
        r["bits_human_seed_counts"],
        yy,
        marker="D",
        s=15,
        color=HUMAN,
        linewidth=0,
        zorder=4,
    )


def plot(df: pl.DataFrame, out_stem: Path) -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7.5,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    rows, headers, y = [], [], 0.0
    for label, lo, hi in SIZE_GROUPS:
        headers.append((y, label))
        y += 0.9
        for r in df.filter(pl.col("n_letters").is_between(lo, hi)).iter_rows(
            named=True
        ):
            rows.append((y, r))
            y += 1.0
        y += 0.5
    assert len(rows) == df.height, "an alphabet fell outside every size group"

    fig = plt.figure(figsize=(7.2, 7.4))
    ax = fig.add_axes([0.21, 0.355, 0.44, 0.505])
    xmax = 4.6
    for x in [0, 1, 2, 3, 4]:
        ax.axvline(x, color=GRID, lw=0.6, zorder=0)
    for yy, r in rows:
        draw_row(ax, yy, r)

    for hy, label in headers:
        ax.text(
            -0.015, hy + 0.15, label, transform=ax.get_yaxis_transform(), ha="right", va="center",
            fontsize=7, fontweight="bold", color=GROUP_INK,
        )  # fmt: skip
    ax.set_yticks([yy for yy, _ in rows])
    ax.set_yticklabels([r["alphabet"] for _, r in rows], fontfamily=MONO, fontsize=6.6)
    ax.set_ylim(y - 0.5, -0.4)
    ax.set_xlim(0, xmax)
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_xlabel("bits of information per letter")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    ax.text(
        1.03, -0.15, "letter groups", transform=ax.get_yaxis_transform(), ha="left", va="bottom",
        fontsize=7, fontweight="bold", color=GROUP_INK,
    )  # fmt: skip
    for yy, r in rows:
        ax.text(
            1.03, yy, letter_groups_text(r["alphabet"], r["letter_groups"].split()),
            transform=ax.get_yaxis_transform(), ha="left", va="center", fontfamily=MONO, fontsize=5.8,
            color="#3a3f47",
        )  # fmt: skip

    # Panel b: the 2-3 letter alphabets on a stretched axis, where the three marks of panel a
    # sit within 0.3 bits of each other.
    hp_rows = [(yy, r) for yy, r in rows if r["n_letters"] <= 3]
    y0 = min(yy for yy, _ in hp_rows) - 0.5
    y1 = max(yy for yy, _ in hp_rows) + 0.5
    lo_b, hi_b = 0.6, 1.65
    ax.add_patch(
        mpl.patches.Rectangle(
            (lo_b, y0),
            hi_b - lo_b,
            y1 - y0,
            fill=False,
            ec=GROUP_INK,
            lw=0.7,
            ls=(0, (3, 2)),
            zorder=5,
        )
    )
    ax.text(
        hi_b + 0.05,
        y0 + 0.1,
        "b",
        fontsize=8,
        fontweight="bold",
        va="top",
        color=GROUP_INK,
    )
    axb = fig.add_axes([0.21, 0.065, 0.44, 0.205])
    for i, (_, r) in enumerate(hp_rows):
        draw_row(axb, i, r)
    axb.set_yticks(range(len(hp_rows)))
    axb.set_yticklabels(
        [r["alphabet"] for _, r in hp_rows], fontfamily=MONO, fontsize=6.6
    )
    axb.set_ylim(len(hp_rows) - 0.5, -0.6)
    axb.set_xlim(lo_b, hi_b)
    axb.set_xticks([0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6])
    for x in axb.get_xticks():
        axb.axvline(x, color=GRID, lw=0.6, zorder=0)
    axb.set_xlabel(
        "bits of information per letter (2–3 letter alphabets only, axis stretched)"
    )
    for side in ("top", "right"):
        axb.spines[side].set_visible(False)
    tr = axb.get_yaxis_transform()
    for x, head in [(1.03, "hydrophobic"), (1.30, "polar"), (1.56, "own letter")]:
        axb.text(
            x,
            -0.75,
            head,
            transform=tr,
            ha="left",
            va="bottom",
            fontsize=7,
            fontweight="bold",
            color=GROUP_INK,
        )
    for i, (_, r) in enumerate(hp_rows):
        for x, letters in zip([1.03, 1.30, 1.56], r["letter_groups"].split()):
            axb.text(
                x,
                i,
                letters,
                transform=tr,
                ha="left",
                va="center",
                fontfamily=MONO,
                fontsize=5.8,
                color="#3a3f47",
            )
    for a, letter in [(ax, "a"), (axb, "b")]:
        fig.text(
            0.02,
            a.get_position().y1 + 0.004,
            letter,
            fontsize=10,
            fontweight="bold",
            va="bottom",
        )

    n_below = df.filter(
        pl.col("bits_human_seed_counts") < pl.col("bits_swissprot_composition")
    ).height
    handles = [
        Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor=NOMINAL, markeredgewidth=1.1,
               markersize=5.5, label="log2(n): the value if all n letters were equally common"),
        Line2D([], [], marker="o", ls="", color=COMPOSITION, markersize=4.2,
               label=f"from letter frequencies in Swiss-Prot {SWISSPROT_RELEASE}"),
        Line2D([], [], marker="D", ls="", color=HUMAN, markersize=3.6,
               label="measured from kmerseek seed counts in the human proteome"),
        Line2D([], [], color=JOIN, lw=1.4, label="joins the three values of one alphabet"),
    ]  # fmt: skip
    many = "all" if n_below == df.height else f"{n_below} of"
    fig.text(
        0.02, 0.975,
        f"Real proteins carry fewer bits per letter than letter counts predict, in {many} {df.height} alphabets",
        fontsize=9, fontweight="bold", va="top",
    )  # fmt: skip
    fig.legend(
        handles=handles, loc="upper left", bbox_to_anchor=(0.015, 0.955), frameon=False, fontsize=6.8,
        handlelength=1.6, handletextpad=0.6, labelspacing=0.35, borderaxespad=0,
    )  # fmt: skip
    for ext in ("png", "pdf", "svg"):
        fig.savefig(f"{out_stem}.{ext}", dpi=600 if ext == "png" else None)


def main() -> None:
    df = build_table()
    with pl.Config(tbl_rows=-1, tbl_cols=-1, float_precision=3, tbl_width_chars=200):
        print(df.drop("letter_groups"))
    n_below = df.filter(
        pl.col("bits_human_seed_counts") < pl.col("bits_swissprot_composition")
    ).height
    print(f"human-measured below composition: {n_below} of {df.height} alphabets")
    df.write_csv(REPO / "tables" / "bits_per_letter_by_alphabet.csv", float_precision=4)
    plot(df, REPO / "figures" / "bits_per_letter_by_alphabet_swissprot_human")


if __name__ == "__main__":
    main()
