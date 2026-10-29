#!/usr/bin/env python3
"""Draw the three figures for the venom-peptide retrieval pilot.

Reads `results/<config>/summary.csv` and `collision_report.csv` and writes to the
repository's `figures/` folder. Run after `make all`:

    python make_figures.py [--results ../../tables/venom_retrieval/pilot]

Colours mean one thing each, across all three figures:
  blue          the target or activity endpoint
  vermillion    the fold endpoint backed by a measured disulfide connectivity
  bluish green  the fold endpoint taken from UniProt's Knottin keyword
  grey          context: chance, and the rest of the population
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "notebooks"))

import pubfig as pf  # noqa: E402

#: One colour per endpoint, used in every figure here.
ENDPOINT_COLOUR = {
    "target": pf.OKABE_ITO["blue"],
    "fold_connectivity": pf.OKABE_ITO["vermillion"],
    "fold_knottin": pf.OKABE_ITO["bluish_green"],
}

#: What each endpoint is, in a biologist's words, for the legend.
ENDPOINT_LABEL = {
    "target": "same measured target or activity",
    "fold_connectivity": "same measured disulfide connectivity",
    "fold_knottin": "both called knottin by UniProt",
}

#: The methods shown, grouped by what they know about the peptide, with a gap between
#: groups. Within a group the order is the ablation ladder: each knows one more thing.
METHOD_GROUPS = [
    (
        "cysteine framework and loops only",
        [
            ("framework_only", "framework, as a label"),
            ("loop_lengths_only", "loop lengths"),
            ("loop_composition_only", "loop composition, order dropped"),
            ("loop_anchored", "loops aligned, loop to loop"),
        ],
    ),
    ("whole peptide", [("full_sequence", "Smith-Waterman, BLOSUM62")]),
    (
        "framework-guided, then reranked by Smith-Waterman",
        [
            ("cysteine_count_then_rerank", "same cysteine count"),
            ("framework_then_rerank", "same count, loops within 2 aa"),
        ],
    ),
    (
        "reduced-alphabet k-mers, every k-mer kept",
        [
            ("kmer_cys2_k12", "cys2 (C vs other), k=12"),
            ("kmer_cys_hp3_k9", "cys_hp3 (C, h, p), k=9"),
        ],
    ),
]

#: Alphabets in the collision figure, grouped by how many classes they have.
ALPHABET_GROUPS = [
    ("2 classes", [("cys2", "cys2: C vs everything else")]),
    (
        "3 classes",
        [
            ("cys_hp3", "cys_hp3: C, hydrophobic, polar"),
            ("hp_pbotc_1st_ed2", "hp_pbotc_1st_ed2: hydrophobic, polar"),
        ],
    ),
    ("6 classes", [("dayhoff6", "dayhoff6")]),
    ("20 classes", [("protein20", "protein20: no reduction")]),
]

#: The two peptides a two-class cysteine alphabet writes as one string. Both are in the
#: manifest; the sequences are repeated here so the figure does not depend on a rebuild.
PAIR = [
    (
        "Conotoxin Mr6.14",
        "Conus marmoreus, Mollusca",
        "SCDQTGEPCVLNEQCCYGWCTNHGTCY",
        "ConoServer P05381",
    ),
    (
        "Nemertide alpha-8",
        "Riseriellus occultus, Nemertea",
        "GCIATGSFCTLSKGCCTKNCGWNFACN",
        "UniProt P0DQT0",
    ),
]


def _rows_with_gaps(groups):
    """y positions for grouped category rows, with one blank row between groups.

    Returns `(positions, labels, group_spans)`, where a span is the (first, last, title)
    of one group, for the group headings.
    """
    positions, labels, spans = [], [], []
    y = 0.0
    for title, members in groups:
        first = y
        for _, label in members:
            positions.append(y)
            labels.append(label)
            y += 1.0
        spans.append((first, y - 1.0, title))
        y += 1.6
    return positions, labels, spans


def _grid(ax, positions):
    """A guide line from each category tick straight across the panel."""
    for y in positions:
        ax.axhline(y, color="#dddddd", linewidth=0.3, zorder=0)


def _category_axis(ax, positions, labels, spans=(), right=False):
    """Set the row ticks, labels, group headings and limits of one panel.

    One function for every panel that shows these rows, so two panels of one figure
    cannot end up with different limits and draw the same row at two different heights.
    `right` puts the labels on the right-hand side, for the second panel of a pair.
    """
    ax.set_yticks(positions)
    if right:
        ax.yaxis.tick_right()
    ax.set_yticklabels(labels)
    ax.set_ylim(positions[-1] + 0.9, positions[0] - 1.9)
    left = ax.get_xlim()[0]
    width = ax.get_xlim()[1] - left
    for first, _, title in spans:
        ax.text(
            left + 0.02 * width, first - 0.78, title, va="center", ha="left", color="#555555"
        )


def figure_precision(summary: pl.DataFrame, out: Path) -> Path:
    """Precision in the top 10, as a multiple of chance, for every endpoint.

    What: method (category, grouped), endpoint (category), precision at the declared
    budget of 10 divided by the chance rate for that endpoint (quantity).
    Why: compare a few methods across three endpoints, and see that the ranking of the
    methods is not the same in all three.
    How: dots on one shared x axis, one row per method, hue per endpoint. The multiple of
    chance rather than the raw precision, because the three endpoints have very different
    base rates and a raw precision would compare them unfairly on one axis.
    """
    held = summary.filter(pl.col("reference_mode") == "held_out_family")
    positions, labels, spans = _rows_with_gaps(METHOD_GROUPS)
    names = [name for _, members in METHOD_GROUPS for name, _ in members]

    fig, ax = pf.figure(pf.TWO_COLUMN_MM, 88)
    _grid(ax, positions)
    ax.axvline(1.0, color=pf.GREY, linewidth=0.5, linestyle="--", zorder=1)
    ax.text(1.15, positions[-1] + 0.55, "chance", color=pf.GREY, va="center", ha="left")

    for endpoint, colour in ENDPOINT_COLOUR.items():
        block = held.filter(pl.col("endpoint") == endpoint)
        by_method = {
            row["method"]: row["mean_precision_at_budget"]
            / (row["mean_chance_tp_at_budget"] / 10)
            for row in block.iter_rows(named=True)
        }
        chance = block["mean_chance_tp_at_budget"][0] / 10
        ax.plot(
            [by_method[name] for name in names],
            positions,
            "o",
            color=colour,
            markersize=3,
            label=f"{ENDPOINT_LABEL[endpoint]} (chance {chance:.3f})",
        )

    ax.set_xlabel("Precision in the top 10, as a multiple of chance (higher is better)")
    ax.set_xlim(0, 13.6)
    # Each group named once above its own rows, not rotated beside them: four rotated
    # titles at the right edge overlapped into one block of letters.
    _category_axis(ax, positions, labels, spans)
    pf.shared_legend(fig, ncol=1)
    stem = out / "venom_retrieval_precision_by_endpoint_swissprot_conoserver"
    pf.save(fig, stem)
    return stem.with_suffix(".png")


def figure_candidate_loss(summary: pl.DataFrame, out: Path) -> Path:
    """Where the framework-guided pipeline loses what it loses.

    What: method (category), the share of true positives a method could compare at all
    (quantity), and recall in the top 10 counted two ways (quantity).
    Why: the framework-guided pipeline reranks well and still loses end to end, and these
    two panels say which step costs it.
    How: panel a, candidate recall per method and endpoint, dots on a shared axis. Panel
    b, the target endpoint only, with recall counted over the true positives the method
    could compare (open circle) and over every true positive in the reference (filled
    circle) on the same row, so the gap between them is the loss at candidate generation.
    """
    held = summary.filter(pl.col("reference_mode") == "held_out_family")
    positions, labels, spans = _rows_with_gaps(METHOD_GROUPS)
    names = [name for _, members in METHOD_GROUPS for name, _ in members]

    fig, axs = pf.figure(pf.TWO_COLUMN_MM, 88, ncols=2)

    _grid(axs[0], positions)
    for endpoint, colour in ENDPOINT_COLOUR.items():
        block = held.filter(pl.col("endpoint") == endpoint)
        by_method = {
            row["method"]: row["mean_candidate_recall"]
            for row in block.iter_rows(named=True)
        }
        axs[0].plot(
            [by_method[name] for name in names],
            positions,
            "o",
            color=colour,
            markersize=3,
            label=ENDPOINT_LABEL[endpoint],
        )
    axs[0].set_xlabel("Share of true positives the method could compare at all")
    axs[0].set_xlim(-0.03, 1.05)
    _category_axis(axs[0], positions, labels, spans)

    target = held.filter(pl.col("endpoint") == "target")
    conditional = {
        row["method"]: row["mean_recall_at_budget_conditional"]
        for row in target.iter_rows(named=True)
    }
    end_to_end = {
        row["method"]: row["mean_recall_at_budget"]
        for row in target.iter_rows(named=True)
    }
    _grid(axs[1], positions)
    for name, y in zip(names, positions):
        axs[1].plot(
            [end_to_end[name], conditional[name]],
            [y, y],
            "-",
            color=pf.GREY,
            linewidth=0.5,
            zorder=1,
        )
    axs[1].plot(
        [conditional[name] for name in names],
        positions,
        "o",
        markerfacecolor="white",
        markeredgecolor=ENDPOINT_COLOUR["target"],
        markersize=3.5,
        label=(
            "panel b, open: recall over the true positives it could compare; "
            "filled: over every true positive"
        ),
    )
    axs[1].plot(
        [end_to_end[name] for name in names],
        positions,
        "o",
        color=ENDPOINT_COLOUR["target"],
        markersize=3,
    )
    axs[1].set_xlabel("Recall in the top 10, same target or activity")
    # A value of 0.0002 would sit under the spine, so the axis starts a little below 0.
    axs[1].set_xlim(-0.012, 0.32)
    _category_axis(axs[1], positions, labels, right=True)
    pf.shared_legend(fig, ncol=2)
    for ax, letter in zip(axs, "ab"):
        pf.panel_label(ax, letter)
    stem = out / "venom_retrieval_candidate_loss_swissprot_conoserver"
    pf.save(fig, stem)
    return stem.with_suffix(".png")


def figure_collisions(collisions: pl.DataFrame, out: Path) -> Path:
    """What a two-class cysteine reading cannot tell apart, counted and shown.

    What: alphabet (ordered category, by how many classes it has), the share of
    cysteine-rich toxins whose whole encoded string is shared with another peptide
    (quantity), and two named peptides drawn residue by residue.
    Why: the count says how often the problem happens; the two peptides say what it looks
    like. One panel without the other is either an abstraction or an anecdote.
    How: panel a, dots on a shared axis, rows grouped by alphabet size. Panel b, each
    peptide as a thin backbone line with a box at every cysteine, the Pfam convention, and
    its cys2 string underneath in a monospaced font.
    """
    cysrich = collisions.filter(pl.col("subset") == "cysteine_rich_toxins")
    positions, labels, _ = _rows_with_gaps(ALPHABET_GROUPS)
    names = [name for _, members in ALPHABET_GROUPS for name, _ in members]
    share = {
        row["alphabet"]: row["share_peptides_in_collision"]
        for row in cysrich.iter_rows(named=True)
    }
    n_peptides = int(cysrich["n_peptides"][0])

    fig, axs = pf.figure(pf.TWO_COLUMN_MM, 62, ncols=2, width_ratios=[1, 1.5])

    _grid(axs[0], positions)
    axs[0].plot(
        [share[name] for name in names],
        positions,
        "o",
        color=pf.OKABE_ITO["vermillion"],
        markersize=3.5,
    )
    for name, y in zip(names, positions):
        axs[0].annotate(
            f"{share[name]:.0%}",
            (share[name], y),
            textcoords="offset points",
            xytext=(5, -2.5),
            color="#555555",
        )
    axs[0].set_xlim(-0.04, 1.0)
    axs[0].set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    axs[0].set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    _category_axis(axs[0], positions, labels)
    axs[0].set_xlabel(
        f"Share of the {n_peptides:_} cysteine-rich toxins whose\n"
        f"whole encoded string another peptide also has\n(lower is better)".replace(
            "_", " "
        )
    )

    ax = axs[1]
    length = len(PAIR[0][2])
    rows = [3.4, 0.9]
    for (name, species, sequence, accession), y in zip(PAIR, rows):
        ax.plot([0.5, length + 0.5], [y, y], "-", color="#333333", linewidth=1.0)
        for index, residue in enumerate(sequence, 1):
            if residue == "C":
                ax.add_patch(
                    pf.plt.Rectangle(
                        (index - 0.42, y - 0.2),
                        0.84,
                        0.4,
                        facecolor=pf.OKABE_ITO["vermillion"],
                        edgecolor="none",
                    )
                )
            # One text call per residue, each centred on its own position. Drawing the
            # sequence as one string put the letters at the font's own spacing, which does
            # not match the residue spacing, so no letter sat under its cysteine box.
            ax.text(
                index,
                y - 0.42,
                residue,
                family="monospace",
                va="top",
                ha="center",
                color=pf.OKABE_ITO["vermillion"] if residue == "C" else "#333333",
            )
        # Name first, then where it comes from, so the panel reads top to bottom.
        ax.text(0.5, y + 0.95, name, va="bottom", ha="left")
        ax.text(
            0.5,
            y + 0.38,
            f"{species} · {accession}",
            va="bottom",
            ha="left",
            color="#555555",
        )

    for index, residue in enumerate(PAIR[0][2], 1):
        ax.text(
            index,
            -0.95,
            "c" if residue == "C" else "x",
            family="monospace",
            va="top",
            ha="center",
            color=pf.OKABE_ITO["vermillion"],
        )
    ax.text(0.5, -0.5, "cys2 writes both peptides as:", va="top", ha="left")

    ax.set_xlim(0, length + 1.5)
    ax.set_ylim(-2.2, 5.4)
    ax.set_xlabel("Residue")
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)

    legend = [
        pf.plt.Line2D([], [], color="#333333", linewidth=1.0, label="peptide backbone"),
        pf.plt.Rectangle(
            (0, 0), 1, 1, facecolor=pf.OKABE_ITO["vermillion"], label="cysteine"
        ),
    ]
    ax.legend(handles=legend, loc="lower right", frameon=False, ncol=2)
    for ax, letter in zip(axs, "ab"):
        pf.panel_label(ax, letter)
    stem = out / "venom_retrieval_cys2_collisions_swissprot_conoserver"
    pf.save(fig, stem)
    return stem.with_suffix(".png")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--results", type=Path, default=REPO / "tables" / "venom_retrieval" / "pilot"
    )
    parser.add_argument("--figures", type=Path, default=REPO / "figures")
    args = parser.parse_args(argv)
    args.figures.mkdir(parents=True, exist_ok=True)

    pf.use_style()
    summary = pl.read_csv(args.results / "summary.csv")
    collisions = pl.read_csv(args.results / "collision_report.csv")
    for drawn in (
        figure_precision(summary, args.figures),
        figure_candidate_loss(summary, args.figures),
        figure_collisions(collisions, args.figures),
    ):
        print(drawn)
    return 0


if __name__ == "__main__":
    sys.exit(main())
