#!/usr/bin/env python3
"""How many of each study's own genes each reference-free route reaches (notebook 361, human).

Usage: human_compare.py <human_genes.tsv> <published_genes.tsv> <gene_basemean.tsv> <out.tsv>

Same null as the Botryllus comparison (design/ANALYSIS_PLAN.md, 2026-10-30): for each contrast,
route and direction, the share of the study's list reached by the route is compared with the
same share in 1,000 random gene sets of the list's size, drawn from the genes of that contrast's
count table (protein-coding, with a GENCODE protein) so that each baseMean decile holds as many
genes as in the list. Seed 361, genes and deciles sorted, one random stream per row.

A route's up list is compared with genes it reached through sequences higher in the second
group, its down list with sequences lower there. SPLASH compactors have no direction and are
compared with both lists alike.
"""
import sys

import numpy as np
import polars as pl

hg, pub, base, out = sys.argv[1:5]
reached = pl.read_csv(hg, separator="\t")
lists = pl.read_csv(pub, separator="\t")
pool_all = pl.read_csv(base, separator="\t")
N_DRAW = 1000
rows = []
for (series, contrast), L in lists.group_by(["series", "contrast"], maintain_order=True):
    pool = (pool_all.filter(pl.col("contrast") == contrast).group_by("gene").agg(pl.col("baseMean").max())
            .sort("gene"))
    if pool.height == 0:
        continue
    q = np.quantile(pool["baseMean"].to_numpy(), np.linspace(0, 1, 11)[1:-1])
    dec = dict(zip(pool["gene"].to_list(), np.searchsorted(q, pool["baseMean"].to_numpy(), side="right").tolist()))
    by_dec = {d: sorted(g for g, x in dec.items() if x == d) for d in range(10)}
    src = L["source"][0]
    R = reached.filter(pl.col("contrast") == contrast)
    for route in sorted(R["route"].unique().to_list()) if R.height else []:
        for direction in ("up", "down"):
            lst = sorted(set(L.filter(pl.col("direction") == direction)["gene"]) & set(dec))
            rr = R.filter((pl.col("route") == route) & pl.col("direction").is_in([direction, "none"]))
            hit = set(rr["gene"].to_list())
            obs = sum(g in hit for g in lst)
            row = dict(series=series, contrast=contrast, list_source=src, route=route, direction=direction,
                       list_genes=len(lst), reached=obs, share=obs / len(lst) if lst else np.nan,
                       route_genes_in_pool=len(hit & set(dec)), null_median=np.nan, null_lo=np.nan,
                       null_hi=np.nan, share_random_at_or_above=np.nan)
            if lst:
                need = np.bincount([dec[g] for g in lst], minlength=10)
                rng = np.random.default_rng([361, len(rows)])
                null = np.empty(N_DRAW)
                for i in range(N_DRAW):
                    pick = []
                    for d in range(10):
                        if need[d]:
                            pick += list(rng.choice(by_dec[d], size=need[d], replace=False))
                    null[i] = sum(g in hit for g in pick) / len(lst)
                row.update(null_median=float(np.median(null)), null_lo=float(np.quantile(null, 0.025)),
                           null_hi=float(np.quantile(null, 0.975)),
                           share_random_at_or_above=float((null >= row["share"] - 1e-12).mean()))
            rows.append(row)
res = pl.DataFrame(rows)
res.write_csv(out, separator="\t")
with pl.Config(tbl_rows=300, tbl_cols=20, tbl_width_chars=250):
    print(res.select("contrast", "route", "direction", "list_genes", "reached", "share", "null_median",
                     "null_hi", "share_random_at_or_above"))
