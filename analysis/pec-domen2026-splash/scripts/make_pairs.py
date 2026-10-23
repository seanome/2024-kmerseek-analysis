#!/usr/bin/env python3
"""Collect sourmash multisearch matches into query-target pairs for the alignment check.

Usage: make_pairs.py <out_pairs.tsv> <out_hits.parquet> <multisearch.csv>...

Each CSV name must end in `.<db>.<moltype>.k<k>.s<scaled>.csv`. A match is kept when the query
and target share at least 2 hashes; per query, setting and database, the 5 matches with the
highest containment are kept (ties broken by the smaller overlap probability, then the target
name). Names are cut to their first word (the stable ID).
"""
import re
import sys

import polars as pl

out_pairs, out_hits, *csvs = sys.argv[1:]
frames = []
for c in csvs:
    m = re.search(r"\.(\w+)\.(protein|dayhoff|hp)\.k(\d+)\.s(\d+)\.csv$", c)
    if not m:
        raise SystemExit(f"cannot parse setting from {c}")
    db, mol, k, s = m.groups()
    df = pl.read_csv(c, infer_schema_length=0)
    if df.height == 0:
        continue
    df = df.with_columns(
        pl.col("query_name").str.split(" ").list.first().alias("query"),
        pl.col("match_name").str.split(" ").list.first().alias("target"),
        pl.col("intersect_hashes").cast(pl.Int64),
        pl.col("containment").cast(pl.Float64),
        pl.lit(db).alias("db"), pl.lit(mol).alias("moltype"), pl.lit(int(k)).alias("ksize"),
        pl.lit(int(s)).alias("scaled"),
    )
    prob = [x for x in df.columns if x.startswith("prob_overlap")]
    df = df.with_columns([pl.col(x).cast(pl.Float64) for x in prob])
    keep = ["query", "target", "db", "moltype", "ksize", "scaled", "containment", "intersect_hashes",
            *prob, "query_name", "match_name"]
    frames.append(df.select([x for x in keep if x in df.columns]))
hits = pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()
if hits.height:
    sort_cols = ["containment"] + (["prob_overlap_adjusted"] if "prob_overlap_adjusted" in hits.columns else [])
    hits = (hits.filter(pl.col("intersect_hashes") >= 2)
            .sort(["query", "db", "moltype", "scaled", *sort_cols, "target"],
                  descending=[False, False, False, False, True] + [False] * (len(sort_cols) - 1) + [False])
            .group_by(["query", "db", "moltype", "scaled"], maintain_order=True).head(5))
hits.write_parquet(out_hits)
pairs = hits.select("query", "target", "db").unique() if hits.height else pl.DataFrame(
    schema={"query": pl.Utf8, "target": pl.Utf8, "db": pl.Utf8})
for db in ("proteome", "swissprot", "botznik"):
    pairs.filter(pl.col("db") == db).select("query", "target").write_csv(
        out_pairs.replace(".tsv", f".{db}.tsv"), separator="\t")
print(f"hits kept {hits.height}, pairs {pairs.height}")
