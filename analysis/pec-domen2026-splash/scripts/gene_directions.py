#!/usr/bin/env python3
"""Up or down per gene from pec-tools gene-counts, summed by group (notebook 361).

Usage: gene_directions.py <counts.tsv> <design.tsv> <out.tsv>

Rule, fixed in exports/cheap_route/README.md before any run. Reads are summed per group, so a
shallow sample weighs little, and each group is scaled by its assigned reads. A gene gets a direction
when it has 10 or more reads in both groups, or 10 or more in one and none in the other, and a
2-fold or larger change. up = higher in PEC. There is no per-sample test.

Writes gene, control_reads, PEC_reads, log2fc (empty when one group has no reads), direction.
"""
import math
import sys

import polars as pl

counts_p, design_p, out = sys.argv[1:]
MIN_READS, MIN_FOLD = 10, 2.0
d = pl.read_csv(design_p, separator="\t", columns=["sample", "condition"])
c = pl.read_csv(counts_p, separator="\t")
groups = {g: d.filter(pl.col("condition") == g)["sample"].to_list() for g in ("control", "PEC")}
assert all(groups.values()), groups
t = c.select("gene", pl.sum_horizontal(groups["control"]).alias("control_reads"), pl.sum_horizontal(groups["PEC"]).alias("PEC_reads"))
total_c, total_p = t["control_reads"].sum(), t["PEC_reads"].sum()
rows = []
for gene, a, b in t.iter_rows():
    if a == 0 and b == 0:
        continue
    fc = math.log2((b / total_p) / (a / total_c)) if a and b else None
    if a >= MIN_READS and b >= MIN_READS and abs(fc) >= math.log2(MIN_FOLD):
        direction = "up" if fc > 0 else "down"
    elif a == 0 and b >= MIN_READS:
        direction = "up"
    elif b == 0 and a >= MIN_READS:
        direction = "down"
    else:
        direction = ""
    rows.append(dict(gene=gene, control_reads=a, PEC_reads=b, log2fc=fc, direction=direction))
r = pl.DataFrame(rows, schema_overrides={"log2fc": pl.Float64})
r.write_csv(out, separator="\t")
print(out, "genes with reads", r.height, "up", (r["direction"] == "up").sum(), "down", (r["direction"] == "down").sum())
