#!/usr/bin/env python3
"""Figure: how the 998 midi-plus queries fall into the query-side strata.

a. Queries per bin of the four protein-level strata (membrane class, SEG low-complexity
   share, protein length, BUSCO status), one bar per bin, bins grouped by stratum.
b. Queries per identity bin to their nearest protein in each of the 15 target proteomes,
   one row per target ordered by time since its line split from ours.

Reads results/leaderboard/query_strata/{query_strata.tsv,query_nearest_hits.tsv} and
prints every count it draws.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl
import seaborn  # noqa: F401  registers the mako colour map

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "notebooks"))
import pubfig as pf  # noqa: E402

AXES = [
    (
        "membrane",
        "Membrane class (DeepTMHMM)",
        ["alpha-helical TM", "beta-barrel", "neither"],
    ),
    ("low_complexity", "Residues masked by SEG", ["<5%", "5-20%", ">20%"]),
    ("protein_length", "Protein length (aa)", ["<100", "100-300", "300-1000", ">1000"]),
    ("busco", "BUSCO, primates_odb12", ["complete", "fragmented", "not a BUSCO gene"]),
]
IDENT = ["no hit", "<30%", "30-50%", "50-90%", ">=90%"]
# Order of the targets: QfO registry divergence from human where it has one (Mya), else
# grouped with their clade (assets/qfo_species.tsv has no time for these six).
TARGETS = [
    ("ptroglodytes", "chimpanzee"),
    ("mouse", "mouse (100 Mya)"),
    ("rnorvegicus", "rat"),
    ("clupus", "dog"),
    ("btaurus", "cow"),
    ("mdomestica", "opossum"),
    ("chicken", "chicken (300)"),
    ("xtropicalis", "frog"),
    ("zebrafish", "zebrafish (430)"),
    ("ciona", "sea squirt (550)"),
    ("fly", "fly (600)"),
    ("worm", "worm (650)"),
    ("yeast", "yeast (900)"),
    ("arabidopsis", "arabidopsis (1500)"),
    ("ecoli", "E. coli (2000)"),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--strata", type=Path, default=Path("results/leaderboard/query_strata")
    )
    args = ap.parse_args()
    q = pl.read_csv(args.strata / "query_strata.tsv", separator="\t")
    h = pl.read_csv(args.strata / "query_nearest_hits.tsv", separator="\t")

    pf.use_style()
    fig, (a, b) = pf.figure(
        pf.TWO_COLUMN_MM, 80, ncols=2, gridspec_kw={"width_ratios": [1, 1.25]}
    )

    # a: one horizontal bar per bin, strata stacked top to bottom with a gap between.
    y, ticks, labels, group_mid = 0, [], [], []
    for axis, title, bins in AXES:
        n = dict(q.group_by(axis).len().iter_rows())
        start = y
        for bn in bins:
            a.barh(y, n.get(bn, 0), color=pf.OKABE_ITO["blue"], height=0.7)
            a.text(n.get(bn, 0) + 8, y, f"{n.get(bn, 0):,}", va="center")
            ticks.append(y)
            labels.append(bn)
            print(f"{axis}\t{bn}\t{n.get(bn, 0)}")
            y += 1
        group_mid.append((start, title))
        y += 1.2
    a.set_yticks(ticks, labels)
    a.invert_yaxis()
    for start, title in group_mid:
        a.text(
            -0.02,
            start - 0.75,
            title,
            transform=a.get_yaxis_transform(),
            ha="right",
            va="bottom",
            fontweight="bold",
        )
    a.set_xlabel(f"Queries (of {q.height:,})")
    a.set_xlim(0, q.height * 1.08)

    # b: heatmap, target x identity bin, counts written in each square.
    counts = dict(
        ((sp, bn), n)
        for sp, bn, n in h.group_by("species", "nearest_identity").len().iter_rows()
    )
    mat = np.array([[counts.get((sp, bn), 0) for bn in IDENT] for sp, _ in TARGETS])
    im = b.imshow(mat, cmap="mako_r", aspect="auto", vmin=0, vmax=mat.max())
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            b.text(
                j,
                i,
                f"{v}",
                ha="center",
                va="center",
                color="white" if v > mat.max() * 0.55 else "black",
            )
    for (sp, _), row in zip(TARGETS, mat):
        print(sp, dict(zip(IDENT, row.tolist())))
    b.set_xticks(
        range(len(IDENT)),
        ["no hit\nE < 0.001", "under\n30%", "30-50%", "50-90%", "90% or\nmore"],
    )
    b.set_yticks(range(len(TARGETS)), [lab for _, lab in TARGETS])
    b.set_xlabel("Identity of the query to its best hit in the target")
    b.set_ylabel("Target proteome (Mya since split from human)")
    cb = fig.colorbar(im, ax=b, fraction=0.05, pad=0.02)
    cb.set_label("Queries")
    pf.panel_label(a, "a", dx_pt=-110)
    pf.panel_label(b, "b", dx_pt=-90)
    out = args.strata / "figures"
    out.mkdir(exist_ok=True)
    pf.save(fig, str(out / "query_strata_counts_midi_plus"))


if __name__ == "__main__":
    main()
