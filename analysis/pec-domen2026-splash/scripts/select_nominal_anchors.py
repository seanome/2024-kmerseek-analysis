#!/usr/bin/env python3
"""Anchors that pass the paper's own cutoff in the anchor-count DESeq2 fit.

Usage: select_nominal_anchors.py <deseq_<design>.parquet> <out.tsv>

Keeps anchors with uncorrected p <= 0.05 and |log2 fold change| > 1, the cutoff the paper used
for its gene lists (nominal P <= 0.05, |log2FC| > 1, SI Dataset S1). Writes anchor first, because
compactors reads the anchor from the first column, then direction (up = higher in PEC),
log2FoldChange, pvalue and baseMean.
"""
import sys

import polars as pl

deseq, out = sys.argv[1:]
d = pl.read_parquet(deseq, columns=["anchor", "baseMean", "log2FoldChange", "pvalue"])
keep = (
    d.filter((pl.col("pvalue") <= 0.05) & (pl.col("log2FoldChange").abs() > 1))
    .with_columns(pl.when(pl.col("log2FoldChange") > 0).then(pl.lit("up")).otherwise(pl.lit("down")).alias("direction"))
    .select("anchor", "direction", "log2FoldChange", "pvalue", "baseMean")
    .sort("anchor")
)
keep.write_csv(out, separator="\t")
print(f"fitted {d.height}, kept {keep.height} "
      f"(up {keep.filter(pl.col('direction') == 'up').height}, down {keep.filter(pl.col('direction') == 'down').height})")
