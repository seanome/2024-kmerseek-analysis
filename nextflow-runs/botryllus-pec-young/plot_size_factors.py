#!/usr/bin/env python3
"""Plot the size-factor test: each of the paper's four comparisons fitted with only the size
factors changed, on this run's counts and on the paper's counts.

Usage: plot_size_factors.py   (run in this folder; reads size_factors_fits.tsv, which
each-alone-checks/size_factors.R writes)
Writes size_factors_summary.tsv (listed genes found over all four comparisons, per fit) and
figures/size_factor_test_domen2026_dataset_s1.{svg,png}.
"""

import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "notebooks"))
import pubfig as pf  # noqa: E402

COMPARISONS = ["2 h", "24 h", "13-49 d", "132 d"]
COUNTS = {"this_run": "This run's counts", "paper": "The paper's counts (Dataset S1)"}
# size factor kind -> (label, colour, marker, row offset)
KINDS = {
    "default": (
        "DESeq2 default, from the same counts",
        pf.OKABE_ITO["blue"],
        "o",
        -0.3,
    ),
    "swapped": (
        "DESeq2 default, from the other data set's counts",
        pf.OKABE_ITO["vermillion"],
        "D",
        -0.15,
    ),
    "poscounts": ("poscounts, from the same counts", pf.OKABE_ITO["bluish_green"], "s", 0.0),
    "total": ("Library total of the same counts", pf.OKABE_ITO["orange"], "^", 0.15),
    # Only on this run's counts: the pipeline's gene-length correction left out.
    "default_no_gene_lengths": (
        "DESeq2 default, from the same counts, no gene-length correction (this run only)",
        pf.OKABE_ITO["reddish_purple"],
        "v",
        0.3,
    ),
}
FOUND = "listed_p_lt_0_05_same_direction"


def summary(fits):
    return (
        fits.group_by("counts", "size_factors", maintain_order=True)
        .agg(
            paper_listed=pl.col("paper_listed").sum(),
            listed_tested=pl.col("listed_tested").sum(),
            listed_found_same_direction=pl.col(FOUND).sum(),
            p_le_0_05_2fold_up=pl.col("p_le_0_05_2fold_up").sum(),
            p_le_0_05_2fold_down=pl.col("p_le_0_05_2fold_down").sum(),
        )
        .with_columns(
            listed_found_pct=(
                100 * pl.col("listed_found_same_direction") / pl.col("paper_listed")
            ).round(1)
        )
    )


def main():
    fits = pl.read_csv("size_factors_fits.tsv", separator="\t")
    assert fits.height == len(COMPARISONS) * (2 * len(KINDS) - 1), fits.height
    total = summary(fits)
    total.write_csv("size_factors_summary.tsv", separator="\t")
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=200):
        print(total)

    plot(total)


# One colour per count table, as in counts_compare and count_split.
COLOUR = {"this_run": pf.OKABE_ITO["blue"], "paper": "black"}
ROW_LABEL = {"this_run": "This run's counts", "paper": "The paper's counts"}
CHOICES = {"default": "DESeq2 default", "poscounts": "poscounts", "total": "library total"}


def plot(total):
    """a: does giving each count table the other's size factors close the gap? b: how much
    the choice of size factors moves each count table on its own. Sums over the four
    comparisons; the per-comparison numbers are in size_factors_fits.tsv and the README."""
    found = {(r["counts"], r["size_factors"]): r["listed_found_same_direction"]
             for r in total.iter_rows(named=True)}
    own_this, own_paper = found["this_run", "default"], found["paper", "default"]
    gap = own_paper - own_this
    pf.use_style()
    fig, (a, b) = pf.figure(pf.TWO_COLUMN_MM, 62, ncols=2, layout="constrained",
                            gridspec_kw=dict(width_ratios=[1.35, 1]))
    rows = {"this_run": 1, "paper": 0}

    a.axvspan(own_this, own_paper, color="#EEEEEE", zorder=0)
    a.text((own_this + own_paper) / 2, 1.55, f"gap between the count tables: {gap} genes",
           ha="center", va="center")
    for counts, y in rows.items():
        c = COLOUR[counts]
        own, swap = found[counts, "default"], found[counts, "swapped"]
        a.annotate("", xy=(swap, y), xytext=(own, y),
                   arrowprops=dict(arrowstyle="-|>", color=c, lw=1, shrinkA=4, shrinkB=4))
        a.plot(own, y, "o", color=c, markersize=5)
        a.plot(swap, y, "o", mfc="white", mec=c, markersize=5, mew=1)
        # One label per row, beside the pair: the right of this run's, the left of the paper's.
        right = counts == "this_run"
        a.annotate(f"{own} → {swap}: {swap - own:+d} genes,\n{100 * abs(swap - own) / gap:.0f} % of the gap",
                   (max(own, swap) if right else min(own, swap), y),
                   xytext=(9 if right else -9, 0), textcoords="offset points",
                   ha="left" if right else "right", va="center", color=c)
    a.plot([], [], "o", color="0.3", markersize=5, label="its own DESeq2 default size factors")
    a.plot([], [], "o", mfc="white", mec="0.3", markersize=5, mew=1,
           label="the other count table's size factors")
    a.legend(loc="lower center", bbox_to_anchor=(0.5, 1.08), ncol=2, frameon=False,
             handletextpad=0.3, columnspacing=1.2)
    a.set_yticks(list(rows.values()), [ROW_LABEL[k] for k in rows])
    a.set_ylim(-0.6, 1.85)
    a.set_xlim(0, 782)
    a.set_xlabel("Paper's listed genes found, of 782\n(p < 0.05, same direction as listed)")

    ys = {k: i for i, k in enumerate(reversed(CHOICES))}
    for y in ys.values():
        b.axhline(y, color="#DDDDDD", linewidth=0.3, zorder=0)
    for counts in rows:
        c = COLOUR[counts]
        for kind, y in ys.items():
            v = found[counts, kind]
            b.plot(v, y, "o", color=c, markersize=5,
                   label=ROW_LABEL[counts] if kind == "default" else None)
            b.annotate(f"{v}", (v, y), xytext=(0, 6), textcoords="offset points", ha="center")
    b.legend(loc="lower center", bbox_to_anchor=(0.5, 1.08), ncol=2, frameon=False,
             handletextpad=0.3, columnspacing=1.2)
    b.set_yticks(list(ys.values()), [CHOICES[k] for k in ys])
    b.set_ylim(-0.6, len(ys) - 0.3)
    b.set_xlim(0, 782)
    b.set_xlabel("Paper's listed genes found, of 782\n(each count table with its own size factors)")
    b.set_ylabel("Size factors")
    for ax, letter in zip((a, b), "ab"):
        pf.panel_label(ax, letter)
    pf.save(fig, "figures/size_factor_test_domen2026_dataset_s1")

if __name__ == "__main__":
    main()
