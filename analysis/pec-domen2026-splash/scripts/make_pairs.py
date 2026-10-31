#!/usr/bin/env python3
"""Collect sourmash manysearch matches into query-target pairs for the alignment check.

Usage: make_pairs.py <out_pairs.tsv> <out_hits.parquet> <manysearch.csv>...

Each CSV name must end in `.<db>.<moltype>.k<k>.s<scaled>.csv`. A match is kept when the query
and target share at least 2 hashes; per query, setting and database, the 5 matches with the
highest containment are kept (ties broken by the smaller overlap probability, then the target
name). Names are cut to their first word (the stable ID).

Each CSV is read in batches and only the running top 5 per query is held, so memory does not
grow with the file. The young_2h compactor search wrote 34 GB of CSV (one Swiss-Prot file was
18 GB), which did not fit in 64 GB when every file was read whole (2026-10-31).
"""
import os
import re
import sys

import polars as pl
import pyarrow.csv as pacsv

KEEP = 5
BLOCK_BYTES = 256 << 20


def top_matches(df, sort_cols):
    """Per query, the KEEP rows with the highest containment, ties by sort_cols then target."""
    by = ["query", *sort_cols, "target"]
    descending = [False, True] + [False] * (len(sort_cols) - 1) + [False]
    top = df.sort(by, descending=descending).group_by("query", maintain_order=True).head(KEEP)
    return top.select(df.columns)  # group_by moves "query" first; keep the order for concat


def read_one(path, db, mol, k, s):
    if os.path.getsize(path) == 0:  # manysearch writes an empty file when nothing matches
        return None
    header = pacsv.open_csv(path, read_options=pacsv.ReadOptions(block_size=1 << 20)).schema.names
    prob = [x for x in header if x.startswith("prob_overlap")]
    cols = ["query_name", "match_name", "containment", "intersect_hashes", *prob]
    sort_cols = ["containment"] + (["prob_overlap_adjusted"] if "prob_overlap_adjusted" in prob else [])
    reader = pacsv.open_csv(
        path,
        read_options=pacsv.ReadOptions(block_size=BLOCK_BYTES),
        convert_options=pacsv.ConvertOptions(
            include_columns=cols,
            column_types={c: "string" for c in cols},
        ),
    )
    best = None
    for batch in reader:
        df = pl.from_arrow(batch)
        if df.height == 0:
            continue
        df = df.with_columns(
            pl.col("intersect_hashes").cast(pl.Int64),
            pl.col("containment").cast(pl.Float64),
            *[pl.col(x).cast(pl.Float64) for x in prob],
        ).filter(pl.col("intersect_hashes") >= 2)
        df = df.with_columns(
            pl.col("query_name").str.split(" ").list.first().alias("query"),
            pl.col("match_name").str.split(" ").list.first().alias("target"),
        )
        best = top_matches(df if best is None else pl.concat([best, df]), sort_cols)
    if best is None or best.height == 0:
        return None
    best = best.with_columns(
        pl.lit(db).alias("db"), pl.lit(mol).alias("moltype"), pl.lit(int(k)).alias("ksize"),
        pl.lit(int(s)).alias("scaled"),
    )
    keep = ["query", "target", "db", "moltype", "ksize", "scaled", "containment", "intersect_hashes",
            *prob, "query_name", "match_name"]
    return best.select(keep), sort_cols


out_pairs, out_hits, *csvs = sys.argv[1:]
frames, sort_cols = [], ["containment"]
for c in csvs:
    m = re.search(r"\.(\w+)\.(protein|dayhoff|hp)\.k(\d+)\.s(\d+)\.csv$", c)
    if not m:
        raise SystemExit(f"cannot parse setting from {c}")
    got = read_one(c, *m.groups())
    if got is not None:
        frames.append(got[0])
        if len(got[1]) > len(sort_cols):
            sort_cols = got[1]
hits = pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()
if hits.height:
    hits = hits.sort(["query", "db", "moltype", "ksize", "scaled", *sort_cols, "target"],
                     descending=[False] * 5 + [True] + [False] * (len(sort_cols) - 1) + [False])
hits.write_parquet(out_hits)
pairs = hits.select("query", "target", "db").unique() if hits.height else pl.DataFrame(
    schema={"query": pl.Utf8, "target": pl.Utf8, "db": pl.Utf8})
for db in ("proteome", "swissprot", "botznik"):
    pairs.filter(pl.col("db") == db).select("query", "target").write_csv(
        out_pairs.replace(".tsv", f".{db}.tsv"), separator="\t")
print(f"hits kept {hits.height}, pairs {pairs.height}")
