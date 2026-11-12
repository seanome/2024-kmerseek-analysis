#!/usr/bin/env python3
"""Human Fig. 4 comparison, one row per route and series (notebook 361).

Usage: human_by_series.py <human_compare_fig4.tsv> <out.tsv>

The three series' reference lists are not made the same way (list_source), so a pooled share mixes
them. Writes genes on the two lists, genes reached in the right direction, genes reached only going
the other way, and lists beating random, for each route and series, and prints it.
"""
import sys

import pandas as pd

in_tsv, out_tsv = sys.argv[1:]
t = pd.read_csv(in_tsv, sep="\t")
t["beat"] = t.share_random_at_or_above <= 0.05
g = t.groupby(["route", "series", "list_source"], as_index=False).agg(list_genes=("list_genes", "sum"), reached=("reached", "sum"),
                                                                      opposite_only=("reached_opposite_only", "sum"),
                                                                      lists_beat_random=("beat", "sum"), lists=("beat", "size"))
g["pct"] = (100 * g.reached / g.list_genes).round(1)
g.to_csv(out_tsv, sep="\t", index=False)
print(g.drop(columns="list_source").to_string(index=False))
