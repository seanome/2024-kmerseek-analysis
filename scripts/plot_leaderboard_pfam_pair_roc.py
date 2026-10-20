#!/usr/bin/env python3
"""Figure: Pfam protein-pair ROC AUC against the target's divergence from human, one line
per tool, read from results/leaderboard/results.parquet (experiment pfam_pair_roc).

Prints every plotted value with its n (positive pairs), so the PR's numbers come from here.
Divergence times are TimeTree 5's (results/leaderboard/timetree_human_pairwise.tsv). A
target TimeTree gives no age for (E. coli: bacteria and human share no dated node) has
no x position, so it is printed but not plotted, and a tool with no other target is
left out of the figure.

    python scripts/plot_leaderboard_pfam_pair_roc.py
"""

import sys
from pathlib import Path

import polars as pl

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "notebooks"))
import pubfig as pf  # noqa: E402

OUT = REPO / "results/leaderboard/figures/pfam_pair_roc_by_divergence"
TOOLS = {
    # tool -> (label, colour)
    "mmseqs2_iterative": ("MMseqs2, 3-iteration profile", pf.OKABE_ITO["blue"]),
    "mmseqs2_seqseq": ("MMseqs2, sequence", pf.OKABE_ITO["sky_blue"]),
    "hmmer3_phmmer": ("phmmer", pf.OKABE_ITO["vermillion"]),
    "diamond": ("DIAMOND ultra-sensitive", pf.OKABE_ITO["bluish_green"]),
}


def main() -> None:
    r = (
        pl.read_parquet(REPO / "results/leaderboard/results.parquet")
        .filter(
            (pl.col("experiment") == "pfam_pair_roc")
            & (pl.col("metric") == "pair_roc_auc")
        )
        .select("tool", "target_db", "target_mya_to_human", "value", "n_instances")
        .sort("tool", "target_mya_to_human")
    )
    with pl.Config(tbl_rows=-1):
        print(r)
    undated = r.filter(pl.col("target_mya_to_human").is_null())
    if undated.height:
        print(
            "not plotted, no TimeTree age:",
            ", ".join(sorted(set(undated["target_db"]))),
            f"({undated.height} rows)",
        )
    r = r.filter(pl.col("target_mya_to_human").is_not_null())
    pf.use_style()
    fig, ax = pf.figure(pf.ONE_COLUMN_MM, 62)
    for tool, (label, colour) in TOOLS.items():
        t = r.filter(pl.col("tool") == tool)
        if t.height == 0:
            print(f"not plotted, no dated target: {tool}")
            continue
        ax.plot(
            t["target_mya_to_human"],
            t["value"],
            marker="o",
            color=colour,
            label=f"{label} ({t.height} species)",
        )
    ax.axhline(0.5, color=pf.GREY, linewidth=0.5)
    ax.text(
        0.98,
        0.505,
        "random ranking",
        ha="right",
        va="bottom",
        color=pf.GREY,
        transform=ax.get_yaxis_transform(),
    )
    ax.set_xlabel("Divergence of the target species from human (million years)")
    ax.set_ylabel("ROC AUC, protein pairs sharing a Pfam domain")
    ax.set_ylim(0.45, 0.9)
    pf.shared_legend(fig, ncol=2)
    pf.save(fig, OUT, formats=("pdf", "png"))
    print(f"wrote {OUT}.png")


if __name__ == "__main__":
    main()
