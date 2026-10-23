"""Figures for the BHF recurrence and CD47 composition checks."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

import pubfig as pf


def recurrence_figure(table: pl.DataFrame, arms: pl.DataFrame, stem: Path) -> None:
    """Compare each listed top protein's frequency for copies and BHF."""
    pf.use_style()
    denominators = dict(arms.group_by("metric").len().iter_rows())
    table = table.sort(
        "metric", "pct_of_copy_arms", "top_gene", descending=[False, True, False]
    )
    labels = [f"{r['metric']}: {r['top_gene']}" for r in table.iter_rows(named=True)]
    bhf = [
        100 * r["n_bhf_arms"] / denominators[r["metric"]]
        for r in table.iter_rows(named=True)
    ]
    fig, ax = pf.figure(pf.TWO_COLUMN_MM, 150)
    y = np.arange(table.height)
    ax.scatter(
        table["pct_of_copy_arms"],
        y - 0.13,
        color=pf.OKABE_ITO["blue"],
        marker="o",
        s=9,
        label="Shuffled copies: rows with this top protein (%)",
    )
    ax.scatter(
        bhf,
        y + 0.13,
        color=pf.OKABE_ITO["vermillion"],
        marker="D",
        s=9,
        label="BHF: alphabet–k pairs with this top protein (%)",
    )
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(left=-0.5)
    ax.set_xlabel("Frequency of a protein as the top hit (%)")
    ax.set_title(
        "kmerseek 0.4: BHF and 300 copies against GENCODE v49 human", loc="left"
    )
    ax.grid(axis="y")
    pf.shared_legend(fig, ncol=1)
    pf.save(fig, stem)
    plt.show()


def hydrophobic_windows_figure(sequences: dict[str, str], stem: Path) -> None:
    """Show every window's residue count, including windows below the cutoff."""
    pf.use_style()
    fig, axs = pf.figure(pf.TWO_COLUMN_MM, 72, ncols=2, sharey=True)
    for ax, gene in zip(axs, ["BCL2", "CD47"]):
        seq = sequences[gene]
        counts = [
            sum(c in "AILMFVW" for c in seq[i : i + 19]) for i in range(len(seq) - 18)
        ]
        ax.plot(
            np.arange(1, len(counts) + 1),
            counts,
            color=pf.OKABE_ITO["blue"],
            label="AILMFVW residues in each 19-aa window",
        )
        ax.axhline(
            14,
            color=pf.OKABE_ITO["vermillion"],
            linestyle="--",
            label="14 residues: notebook composition cutoff",
        )
        ax.set_xlabel(f"{gene} window start (aa)")
        ax.set_title("GENCODE v49 sequence composition", loc="left")
        ax.set_ylim(0, 19)
    axs[0].set_ylabel("AILMFVW residues (n)")
    pf.shared_legend(fig, ncol=1)
    pf.save(fig, stem)
    plt.show()


def membrane_alignment_figure(matches: pl.DataFrame, stem: Path) -> None:
    """Show the first three matches by frequency for each tested membrane alphabet."""
    import bhf_feature_utils as features

    pf.use_style()
    fig, axs = pf.figure(pf.TWO_COLUMN_MM, 120, ncols=2)
    for ax, alphabet in zip(axs, ["gbmr7", "polarity4"]):
        ax.set_axis_off()
        table = features.class_agreement(
            matches.filter(pl.col("alphabets").list.contains(alphabet)).head(3),
            alphabet,
        )
        classes = "; ".join(
            f"{i}={s}" for i, s in enumerate(features.CLUSTERS[alphabet])
        )
        ax.text(
            0,
            1.02,
            f"{alphabet}\n{classes}\nBlue: BHF; orange: human. | = agreement",
            va="bottom",
            transform=ax.transAxes,
            wrap=True,
        )
        for i, row in enumerate(table.iter_rows(named=True)):
            q, t = row["bhf_residues"], row["human_residues"]
            qe, te = row["bhf_encoded"], row["human_encoded"]
            if len(q) != len(t):
                raise ValueError("Ungapped alignment has unequal residue lengths")
            raw = "".join("|" if a == b else " " for a, b in zip(q, t))
            encoded = "".join("|" if a == b else " " for a, b in zip(qe, te))
            label = (
                f"BHF {row['region_start'] + 1}-{row['region_end']} / "
                f"{row['gene']} {row['target_start'] + 1}-{row['target_end']}\n"
                f"{row['identical']}/{len(q)} identical; {row['same_class']}/{len(q)} same class"
            )
            ax.text(0, 0.93 - i * 0.32, label, va="top", transform=ax.transAxes)
            for line, (label, sequence, colour) in enumerate(
                [
                    ("BHF  ", q, pf.OKABE_ITO["blue"]),
                    ("     ", raw, "black"),
                    ("Human", t, pf.OKABE_ITO["vermillion"]),
                    ("BHF  ", qe, pf.OKABE_ITO["blue"]),
                    ("     ", encoded, "black"),
                    ("Human", te, pf.OKABE_ITO["vermillion"]),
                ]
            ):
                ax.text(
                    0,
                    0.85 - i * 0.32 - line * 0.024,
                    label + " " + sequence,
                    va="top",
                    transform=ax.transAxes,
                    family="monospace",
                    color=colour,
                )
    fig.suptitle(
        "kmerseek 0.4: BHF versus GENCODE v49 human membrane regions; first three by frequency"
    )
    pf.save(fig, stem)
    plt.show()
