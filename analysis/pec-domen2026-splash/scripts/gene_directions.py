#!/usr/bin/env python3
"""Up or down per gene from pec-tools gene-counts, summed by group (notebook 361).

Usage: gene_directions.py <counts.tsv> <design.tsv> <out.tsv> [sample_mean | size_factor_mean]

Rule, fixed in exports/cheap_route/README.md before any run. Reads are summed per group, so a
shallow sample weighs little, and each group is scaled by its assigned reads. A gene gets a direction
when it has 10 or more reads in both groups, or 10 or more in one and none in the other, and a
2-fold or larger change. up = higher in PEC. There is no per-sample test.

sample_mean is a second rule, added 2026-11-05 after the summed rule was scored. One sample held up
to 51 % of a contrast's reads, so the sum followed that sample. Here each sample is scaled to reads
per million assigned reads and each group takes the mean of its samples. A gene gets a direction with
10 or more reads summed in at least one group and a 2-fold or larger change between the group means.
log2fc is then between the means.

size_factor_mean is a third rule, planned 2026-11-10 in design/sample_scaling/README.md before it was
scored: the sample_mean rule, with each sample divided by its DESeq2 poscounts size factor (taken over
the control and PEC samples named in the design file) in place of its total. Cutoffs are the same.

Writes gene, control_reads, PEC_reads, log2fc (empty when one group has no reads), direction.
"""
import math
import sys

import polars as pl

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from pec_analysis_utils import scaled_per_million  # noqa: E402

counts_p, design_p, out = sys.argv[1:4]
SIZE_FACTOR = sys.argv[4:] == ["size_factor_mean"]
SAMPLE_MEAN = SIZE_FACTOR or sys.argv[4:] == ["sample_mean"]
assert SAMPLE_MEAN or len(sys.argv) == 4, sys.argv
MIN_READS, MIN_FOLD = 10, 2.0
d = pl.read_csv(design_p, separator="\t", columns=["sample", "condition"])
c = pl.read_csv(counts_p, separator="\t")
groups = {g: d.filter(pl.col("condition") == g)["sample"].to_list() for g in ("control", "PEC")}
assert all(groups.values()), groups
t = c.select("gene", pl.sum_horizontal(groups["control"]).alias("control_reads"), pl.sum_horizontal(groups["PEC"]).alias("PEC_reads"))
total_c, total_p = t["control_reads"].sum(), t["PEC_reads"].sum()
rows = []
if SAMPLE_MEAN:
    cols = [s for g in groups.values() for s in g]
    if SIZE_FACTOR:
        means = pl.DataFrame(scaled_per_million(c.select(cols).to_numpy()), schema=cols)
    else:
        means = c.select(*[(pl.col(s) / pl.col(s).sum() * 1e6).alias(s) for s in cols])
    t = t.with_columns(means.select(pl.mean_horizontal(groups["control"]).alias("control_mean"),
                                    pl.mean_horizontal(groups["PEC"]).alias("PEC_mean")))
    for gene, a, b, ma, mb in t.iter_rows():
        if a == 0 and b == 0:
            continue
        fc = math.log2(mb / ma) if a and b else None
        changed = fc is None or abs(fc) >= math.log2(MIN_FOLD)
        up = b > 0 and (a == 0 or fc > 0)
        rows.append(dict(gene=gene, control_reads=a, PEC_reads=b, log2fc=fc,
                         direction=("up" if up else "down") if max(a, b) >= MIN_READS and changed else ""))
else:
    for gene, a, b, *_ in t.iter_rows():
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
