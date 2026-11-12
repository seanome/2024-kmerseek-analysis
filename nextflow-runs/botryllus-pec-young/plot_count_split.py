#!/usr/bin/env python3
"""Plot the test of which part of the counts causes the gap with the paper's gene lists.

Usage: plot_count_split.py   (run in this folder; reads count_split_fits.tsv,
count_split_listed.tsv and count_split_libraries.tsv, which each-alone-checks/count_split.R
writes through count_split.nf, and each-alone-comparison/paper/paper_gene_overlap_genes.tsv)
Writes count_split_summary.tsv (one row per count table), count_split_match.tsv (the paper's
found genes split by whether they have a matched, tested gene in this run) and
figures/count_split_domen2026_dataset_s1.{svg,png}.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "notebooks"))
import pubfig as pf  # noqa: E402

COMPARISONS = ["2 h", "24 h", "13-49 d", "132 d"]
FOUND = "listed_p_lt_0_05_same_direction"
# Each listed gene with its matched gene in this run and how close the protein match is.
MATCHES = "each-alone-comparison/paper/paper_gene_overlap_genes.tsv"
PAPER, THIS = pf.OKABE_ITO["vermillion"], pf.OKABE_ITO["blue"]
# Rows of panels a and b, top to bottom: (label, genome, counter, marker). Colour is the
# genome, a filled mark is counting as HTSeq does, an open mark is Salmon. The marker is the
# reads: a circle for this run's reads (trimmed with a 62-base minimum length, each pair
# counted once), a square for the same with each mate counted as a read of its own, a
# triangle for the reads trimmed again with a 20-base minimum length.
ROWS = [
    ("The paper's own table", "paper_table", None, "D"),
    ("Paper's genome, as HTSeq does, trimmed at 20 bases", "paper_genome_length20", "featurecounts", "^"),
    ("Paper's genome, by Salmon, trimmed at 20 bases", "paper_genome_length20", "salmon", "^"),
    ("Paper's genome, as HTSeq does, each mate counted", "paper_genome", "featurecounts_reads", "s"),
    ("Paper's genome, counted as HTSeq does", "paper_genome", "featurecounts", "o"),
    ("Paper's genome, counted by Salmon", "paper_genome", "salmon", "o"),
    ("This run's genome, as HTSeq does, each mate counted", "this_genome", "featurecounts_reads", "s"),
    ("This run's genome, counted as HTSeq does", "this_genome", "featurecounts", "o"),
    ("This run's genome, counted by Salmon", "this_genome", "salmon", "o"),
]


def strand_used(libraries):
    """The featureCounts strand setting whose counted pairs per library are closest to the
    paper's on the paper's genome; strand ignored (0) until that table exists."""
    on_paper = libraries.filter(pl.col("table").str.contains(r"^paper_genome__featurecounts_s\d$"))
    if on_paper.is_empty():
        return 0
    share = (
        on_paper.with_columns(share=pl.col("counted") / pl.col("paper_counted"))
        .group_by("table")
        .agg(pl.col("share").median())
        .with_columns(off=(pl.col("share") - 1).abs())
        .sort("off")
    )
    return int(share["table"][0][-1])


def table_name(genome, counter, strand):
    if counter is None:
        return genome
    return f"{genome}__{counter}" + (f"_s{strand}" if counter.startswith("featurecounts") else "")


def main():
    fits = pl.read_csv("count_split_fits.tsv", separator="\t")
    listed = pl.read_csv("count_split_listed.tsv", separator="\t", null_values=["NA"])
    libraries = pl.read_csv("count_split_libraries.tsv", separator="\t", null_values=["NA"])
    strand = strand_used(libraries)

    summary = (
        fits.group_by("table", maintain_order=True)
        .agg(
            genes=pl.col("genes").first(),
            paper_listed=pl.col("paper_listed").sum(),
            listed_with_a_gene=pl.col("listed_with_a_gene").sum(),
            listed_tested=pl.col("listed_tested").sum(),
            listed_found_same_direction=pl.col(FOUND).sum(),
            p_le_0_05_2fold_up=pl.col("p_le_0_05_2fold_up").sum(),
            p_le_0_05_2fold_down=pl.col("p_le_0_05_2fold_down").sum(),
        )
        .join(
            libraries.with_columns(share=pl.col("counted") / pl.col("paper_counted"))
            .group_by("table")
            .agg(
                counted_pairs=pl.col("counted").sum(),
                median_counted_share_of_paper_table=pl.col("share").median().round(3),
                median_spearman_with_paper_table=pl.col("spearman_with_paper_table").median(),
            ),
            on="table",
        )
        .with_columns(strand_setting_read=pl.lit(strand))
    )
    summary.write_csv("count_split_summary.tsv", separator="\t")

    # The paper's found genes, by what this run's own table (Salmon, this genome) has for them.
    wide = listed.filter(pl.col("table") == "paper_table").select(
        "comparison", "paper_gene", "direction", paper_found="found"
    ).join(
        listed.filter(pl.col("table") == "this_genome__salmon").select(
            "comparison", "paper_gene", "direction", "gene", "tested", "log2FoldChange", "found"
        ),
        on=["comparison", "paper_gene", "direction"],
    )
    assert wide.height == 782, wide.height
    on_paper = wide.filter(pl.col("paper_found"))
    match = pl.DataFrame(
        {
            "found_on_paper_table": [
                "found in this run too",
                "tested in this run, not found",
                "matched gene, not tested in this run",
                "no matched gene in this run",
            ],
            "listed_genes": [
                on_paper.filter(pl.col("found")).height,
                on_paper.filter(pl.col("tested") & ~pl.col("found")).height,
                on_paper.filter(pl.col("gene").is_not_null() & ~pl.col("tested")).height,
                on_paper.filter(pl.col("gene").is_null()).height,
            ],
        }
    )
    assert match["listed_genes"].sum() == on_paper.height
    match.write_csv("count_split_match.tsv", separator="\t")
    # The paper's found genes that are tested in this run: how many have a fold change in
    # the listed direction here, and how many are found, by how close the protein match is.
    tested = on_paper.filter(pl.col("tested")).join(
        pl.read_csv(MATCHES, separator="\t", null_values=["NA", ""]).select(
            "comparison", "paper_gene", "direction", "pident", "coverage"
        ),
        on=["comparison", "paper_gene", "direction"],
        how="left",
    )
    assert tested.height == on_paper.filter(pl.col("tested")).height
    assert tested["pident"].null_count() == 0
    same_direction = tested.filter(
        pl.when(pl.col("direction") == "up")
        .then(pl.col("log2FoldChange") > 0)
        .otherwise(pl.col("log2FoldChange") < 0)
    ).height
    closest = (pl.col("pident") >= 99) & (pl.col("coverage") >= 90)
    close = (pl.col("pident") >= 95) & (pl.col("coverage") >= 80)
    by_match = pl.DataFrame(
        [
            {"protein_match": label, "found": t.filter(pl.col("found")).height, "tested": t.height}
            for label, t in (
                ("99% identity over 90% or better", tested.filter(closest)),
                ("95% over 80% or better, under 99% over 90%", tested.filter(close & ~closest)),
                ("under 95% over 80%", tested.filter(~close)),
            )
        ]
    )
    assert by_match["tested"].sum() == tested.height
    # Counted as HTSeq does against counted by Salmon, on this run's genes: the listed genes
    # found on both tables and on one only.
    as_htseq = table_name("this_genome", "featurecounts", strand)
    both_ways = wide.join(
        listed.filter(pl.col("table") == as_htseq).select(
            "comparison", "paper_gene", "direction", found_as_htseq="found"
        ),
        on=["comparison", "paper_gene", "direction"],
    )
    assert both_ways.height == wide.height
    on_both = both_ways.filter(pl.col("found_as_htseq") & pl.col("found")).height
    htseq_only = both_ways.filter(pl.col("found_as_htseq") & ~pl.col("found")).height
    salmon_only = both_ways.filter(~pl.col("found_as_htseq") & pl.col("found")).height
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=250, fmt_str_lengths=60):
        print(summary)
        print(match)
        print("featureCounts strand setting read:", strand)
        print(
            f"found on the paper's table and tested in this run: {tested.height}; "
            f"fold change in the listed direction in this run: {same_direction}"
        )
        print(by_match)
        print(
            "found on this_genome__salmon and not on the paper's table:",
            wide.filter(pl.col("found") & ~pl.col("paper_found")).height,
            "; found on the paper's table and not on this_genome__salmon:",
            wide.filter(pl.col("paper_found") & ~pl.col("found")).height,
        )
        print(
            f"listed genes found on {as_htseq} and on this_genome__salmon: {on_both}; "
            f"only as HTSeq does: {htseq_only}; only by Salmon: {salmon_only}"
        )

    pf.use_style()
    # Panel a is on top and names every mark, so it is the key for panel b under it.
    fig = plt.figure(figsize=(pf.TWO_COLUMN_MM * pf.MM, 120 * pf.MM), layout="constrained")
    axs = fig.subplot_mosaic([["a", "a"], ["b", "c"]], height_ratios=[1.5, 1])
    left, mid, right = axs["a"], axs["b"], axs["c"]
    have = set(fits["table"])
    y = np.arange(len(ROWS))
    offsets = np.linspace(-0.3, 0.3, len(ROWS))
    for i, (label, genome, counter, marker) in enumerate(ROWS):
        name = table_name(genome, counter, strand)
        colour = "black" if counter is None else (PAPER if genome.startswith("paper_genome") else THIS)
        style = dict(
            marker=marker,
            color=colour,
            markerfacecolor="white" if counter == "salmon" else colour,
            linestyle="",
        )
        if name not in have:
            # Not run yet: a small grey x, which no count table uses.
            left.plot(4, i, marker="x", color=pf.GREY, markersize=2.5, linestyle="")
            left.annotate("not run yet", (4, i), xytext=(5, 0), textcoords="offset points",
                          va="center", color="black")
            continue
        t = fits.filter(pl.col("table") == name)
        total = t[FOUND].sum()
        left.plot(total, i, **style)
        left.annotate(str(total), (total, i), xytext=(5, 0), textcoords="offset points", va="center")
        for j, c in enumerate(COMPARISONS):
            (r,) = t.filter(pl.col("comparison") == c).iter_rows(named=True)
            mid.plot(100 * r[FOUND] / r["paper_listed"], j + offsets[i], **style)
    left.set_yticks(y, [r[0] for r in ROWS])
    left.invert_yaxis()
    left.set_xlim(0, 782)
    left.set_xlabel("Paper's listed genes found,\np < 0.05 in the listed direction (n of 782)")
    left.set_ylabel("Count table, same fit for all")
    mid.set_yticks(np.arange(len(COMPARISONS)), COMPARISONS)
    mid.invert_yaxis()
    mid.set_xlim(0, 90)
    mid.set_xlabel("Listed genes found, per comparison (%),\nmarks as in panel a")
    mid.set_ylabel("Paper comparison")
    for ax, n in ((left, len(ROWS)), (mid, len(COMPARISONS))):
        for i in range(n):
            ax.axhline(i, color="#DDDDDD", linewidth=0.3, zorder=0)

    # Panel c: the genes found on the paper's table, by what this run has for them.
    shades = [THIS, "#9CC7E3", pf.GREY, "#DDDDDD"]
    for k, ((what, n), shade) in enumerate(zip(match.iter_rows(), shades)):
        right.barh(k, n, color=shade, height=0.6)
        right.annotate(str(n), (n, k), xytext=(3, 0), textcoords="offset points", va="center")
    right.set_yticks(range(match.height), match["found_on_paper_table"].to_list())
    right.invert_yaxis()
    right.set_xlim(0, 400)
    right.set_xlabel(
        f"Listed genes (n of the {match['listed_genes'].sum()} found\n"
        "on the paper's own table)"
    )
    for ax, letter in zip((left, mid, right), "abc"):
        pf.panel_label(ax, letter)
    pf.save(fig, "figures/count_split_domen2026_dataset_s1")


if __name__ == "__main__":
    main()
