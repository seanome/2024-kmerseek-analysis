#!/usr/bin/env python3
"""How many genes does DESeq2's default size factor (median of ratios) rest on, per comparison?

It uses only genes counted in every library of the fit. The paper-style refit
(differentialabundance-paper/) fits each comparison on its own libraries, so the count is per
comparison, for the paper's counts (Dataset S1) and for this run's counts.

Usage: size_factor_genes.py <paper_counts.tsv> <paper_groups.tsv>
           <salmon.merged.gene_counts.tsv> <samples_paper.csv>
"""

import sys

import polars as pl

# Paper group number -> this run's window for the same comparison.
GROUPS = {1: "young_2h", 2: "young_24h", 4: "young_13_49d", 5: "young_132d"}


def counted_everywhere(counts, columns):
    # DESeq2 rounds counts first (salmon gives fractions), so a gene needs a count of 1 after rounding.
    return (
        counts.select(columns)
        .filter(pl.all_horizontal(pl.all().cast(pl.Float64).round(0) > 0))
        .height
    )


def main(paper_counts, paper_groups, run_counts, samples):
    paper = pl.read_csv(
        paper_counts, separator="\t", null_values=["NA", ""]
    ).drop_nulls()
    groups = pl.read_csv(paper_groups, separator="\t")
    ours = pl.read_csv(run_counts, separator="\t", infer_schema_length=None)
    sheet = pl.read_csv(samples)
    rows = []
    for k, window in GROUPS.items():
        vials = groups.filter(pl.col("group") == k)["vial"].to_list()
        mine = sheet.filter(pl.col("group").str.starts_with(window + "_"))
        assert sorted(vials) == sorted(mine["vial"]), window
        rows.append(
            {
                "comparison": window,
                "libraries": len(vials),
                "paper_genes": paper.height,
                "paper_counted_in_every_library": counted_everywhere(paper, vials),
                "this_run_genes": ours.height,
                "this_run_counted_in_every_library": counted_everywhere(
                    ours, mine["sample"].to_list()
                ),
            }
        )
    with pl.Config(tbl_cols=-1, tbl_width_chars=200):
        print(pl.DataFrame(rows))


if __name__ == "__main__":
    main(*sys.argv[1:5])
