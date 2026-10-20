#!/usr/bin/env python3
"""PR 160's headline figure: Fmax of the best-hit rows beside the main run's rows.

Rows of the figure, top to bottom:
  - each QfO relative of the midi-plus run (PR 148's results.parquet,
    experiment qfo_region_midi_plus): phmmer and MMseqs2 seq-seq -s 7, every hit, whole
    aligned region as the call, with and without merging duplicate calls;
  - Swiss-Prot 2026_03 minus Mammalia, one row per PR 160 setting: the best hit, scored
    with the main run's call (whole aligned region) and with PR 160's call (the hit's
    domains placed through the alignment), from call_rules.parquet;
  - hmmscan against Pfam-A, the ceiling.
All rows: 998 human midi-plus queries, Pfam-A truth, split "all", stratum "all".
Higher Fmax is better; 1 is every true domain found and nothing else.

    python scripts/annotator_g1_figure.py --pr148 <results.parquet from PR 148> \
        --call-rules results/leaderboard/annotator_tools/call_rules.parquet \
        --rows results/leaderboard/annotator_tools/rows.parquet
"""

import argparse
import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks"))
import pubfig as pf  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

OUT = ROOT / "figures/annotator_tools/fmax_best_hit_vs_qfo_relatives_midi_plus"
COLOUR = {
    "phmmer": pf.OKABE_ITO["orange"],
    "MMseqs2": pf.OKABE_ITO["blue"],
    "DIAMOND": pf.OKABE_ITO["bluish_green"],
    "hmmscan": "#000000",
}
# Marker = how the call is made. Filled vs open separates "duplicates kept" from
# "duplicates merged"; shape separates every hit / best hit / placed domain.
RULE = {
    "every hit, whole aligned region": dict(marker="o", fill=True),
    "every hit, whole aligned region, duplicates merged": dict(marker="o", fill=False),
    "best hit, whole aligned region": dict(marker="^", fill=True),
    "best hit, Pfam domain placed through the alignment": dict(marker="s", fill=True),
    "Pfam-A models on the query (hmmscan)": dict(marker="D", fill=True),
}
NAME = {
    "ptroglodytes": "chimpanzee",
    "rnorvegicus": "rat",
    "btaurus": "cow",
    "clupus": "dog",
    "mdomestica": "opossum",
    "xtropicalis": "Xenopus",
    "ciona": "Ciona",
    "fly": "Drosophila",
    "worm": "C. elegans",
    "arabidopsis": "Arabidopsis",
    "ecoli": "E. coli",
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pr148", type=Path, required=True)
    ap.add_argument("--call-rules", type=Path, required=True)
    ap.add_argument("--rows", type=Path, required=True)
    a = ap.parse_args()

    r = pl.read_parquet(a.pr148).filter(
        (pl.col("experiment") == "qfo_region_midi_plus")
        & (pl.col("metric") == "fmax")
        & (pl.col("stratum_axis") == "all")
        & (pl.col("truth_source") == "Pfam-A domain instances")
        & pl.col("tool").is_in(["hmmer3_phmmer", "mmseqs2_seqseq", "hmmscan"])
    )
    qfo = r.filter(pl.col("tool") != "hmmscan").sort("target_mya_to_human", "target_db")
    hmmscan = r.filter(pl.col("tool") == "hmmscan")
    rules = pl.read_parquet(a.call_rules)
    ours = pl.read_parquet(a.rows).filter(
        (pl.col("metric") == "fmax") & (pl.col("stratum_axis") == "all")
    )

    # Rows: (label, n, [(x, colour, rule)])
    rows = []
    for (db, mya), g in qfo.group_by(
        ["target_db", "target_mya_to_human"], maintain_order=True
    ):
        sp = db.removeprefix("QfO ")
        pts = []
        for t in g.iter_rows(named=True):
            tool = "phmmer" if t["tool"] == "hmmer3_phmmer" else "MMseqs2"
            rule = (
                "every hit, whole aligned region"
                if t["dedup_or_raw"] == "raw"
                else "every hit, whole aligned region, duplicates merged"
            )
            pts.append((t["value"], COLOUR[tool], rule))
        n = g["n_instances"].unique().to_list()
        rows.append((f"QfO {NAME.get(sp, sp)} ({mya:g} Mya)", n, pts))
    setting_label = {
        ("diamond_besthit", "sensitive"): ("DIAMOND --sensitive", "DIAMOND"),
        ("diamond_besthit", "ultra-sensitive"): (
            "DIAMOND --ultra-sensitive",
            "DIAMOND",
        ),
        ("mmseqs2_besthit", "s7.5"): ("MMseqs2 -s 7.5", "MMseqs2"),
    }
    for (tool, setting), (label, colour_key) in setting_label.items():
        g = rules.filter((pl.col("tool") == tool) & (pl.col("setting") == setting))
        pts = []
        for t in g.iter_rows(named=True):
            rule = (
                "best hit, whole aligned region"
                if t["call_rule"] == "whole_region"
                else "best hit, Pfam domain placed through the alignment"
            )
            pts.append((t["fmax"], COLOUR[colour_key], rule))
        # The placed value must be the pipeline's own row.
        row = ours.filter((pl.col("tool") == tool) & (pl.col("setting") == setting))
        placed = g.filter(pl.col("call_rule") == "placed")["fmax"][0]
        assert abs(row["value"][0] - placed) < 1e-12, (tool, setting)
        rows.append(
            (
                f"Swiss-Prot minus Mammalia, {label}",
                g["n_truth_instances"].unique().to_list(),
                pts,
            )
        )
    h = hmmscan.row(0, named=True)
    rows.append(
        (
            "Pfam-A library (hmmscan, the ceiling)",
            [h["n_instances"]],
            [(h["value"], COLOUR["hmmscan"], "Pfam-A models on the query (hmmscan)")],
        )
    )

    for label, n, pts in rows:
        print(
            f"{label:58s} n={n} " + "  ".join(f"{x:.3f} [{rule}]" for x, _, rule in pts)
        )

    pf.use_style()
    fig, ax = pf.figure(pf.TWO_COLUMN_MM, 120)
    # Gaps between the three groups: 15 QfO rows, 3 Swiss-Prot rows, the ceiling.
    n_qfo = qfo["target_db"].n_unique()
    y, ys = 0.0, []
    for i in range(len(rows)):
        if i in (n_qfo, n_qfo + 3):
            y += 0.8
        ys.append(y)
        y += 1
    for yy, (label, n, pts) in zip(ys, rows):
        ax.axhline(yy, color="#dddddd", lw=0.3, zorder=0)
        for x, colour, rule in pts:
            st = RULE[rule]
            ax.scatter(
                x,
                yy,
                s=22,
                marker=st["marker"],
                linewidths=0.8,
                facecolors=colour if st["fill"] else "white",
                edgecolors=colour,
                zorder=3,
            )
        ax.text(
            1.005,
            yy,
            f"n = {n[0]:,}" if len(n) == 1 else f"n = {n}",
            transform=ax.get_yaxis_transform(),
            va="center",
            ha="left",
        )
    ax.set_yticks(ys, [r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Fmax over Pfam domains on 998 human proteins (higher is better)")
    ax.text(
        1.005,
        -0.9,
        "truth instances",
        transform=ax.get_yaxis_transform(),
        va="bottom",
        ha="left",
    )
    # Two keys above the plot: colour = search tool (drawn as a line, so it is not
    # read as a shape), shape and fill = how the call is made (drawn in grey).
    tools = [Line2D([], [], color=c, lw=3, label=k) for k, c in COLOUR.items()]
    shapes = [
        Line2D(
            [],
            [],
            ls="",
            marker=st["marker"],
            markersize=5,
            markerfacecolor="#555555" if st["fill"] else "white",
            markeredgecolor="#555555",
            label=k,
        )
        for k, st in RULE.items()
    ]
    ax.add_artist(
        ax.legend(
            handles=tools,
            title="Colour: search tool",
            loc="lower left",
            bbox_to_anchor=(-0.42, 1.01),
            ncol=2,
            frameon=False,
            alignment="left",
        )
    )
    ax.legend(
        handles=shapes,
        title="Shape: how the call is made",
        loc="lower left",
        bbox_to_anchor=(0.12, 1.01),
        ncol=2,
        frameon=False,
        alignment="left",
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pf.save(fig, str(OUT))
    print(f"wrote {OUT}.png")


if __name__ == "__main__":
    main()
