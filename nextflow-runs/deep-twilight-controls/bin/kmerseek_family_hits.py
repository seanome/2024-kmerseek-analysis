#!/usr/bin/env python3
"""Keep the kmerseek regions whose query and target are two different family proteins.

The search runs against the whole human proteome so that the E-values mean what they say;
only the family-to-family regions are scored. Every column kmerseek wrote is kept, typed.

    kmerseek_family_hits.py search.csv family.fasta alphabet ksize extended out.parquet
"""

import sys

import polars as pl

csv, family_fasta, alphabet, ksize, extended, out = sys.argv[1:7]
family = [
    line[1:].split()[0].split("|")[1]
    for line in open(family_fasta)
    if line.startswith(">")
]

acc = lambda col: pl.col(col).str.split(" ").list.get(0).str.split("|").list.get(1)
hits = (
    pl.read_csv(csv, infer_schema_length=0)
    .with_columns(query=acc("query_name"), target=acc("target_name"))
    .filter(pl.col("target").is_in(family) & (pl.col("query") != pl.col("target")))
)
numeric = [
    c
    for c in hits.columns
    if c.startswith(("region_", "target_start", "target_end", "containment", "query_"))
    and c
    not in ("region_subseq", "region_evalue_source", "query_name", "query_md5", "query")
]
hits = hits.with_columns(
    [pl.col(c).cast(pl.Float64, strict=False) for c in numeric],
    alphabet=pl.lit(alphabet),
    ksize=pl.lit(int(ksize)),
    extended=pl.lit(extended == "true"),
)
hits.write_parquet(out)
print(f"{alphabet} k{ksize}: {hits.height} family-to-family regions")
