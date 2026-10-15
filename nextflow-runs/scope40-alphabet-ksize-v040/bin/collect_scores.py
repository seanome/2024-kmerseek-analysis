#!/usr/bin/env python3
"""Stack every setting's AUC table into one.

  collect_scores.py <expected_searches.tsv> <out.tsv> <setting tables...>

Settings that were refused (nofit) or found nothing (empty) have a row with only their status.
A search the run was asked for that has no table at all (its index or search failed three
times and was dropped) gets a row with status `failed`. So the stacked table names every
search, not only the ones that scored.
"""

import sys

import polars as pl

expected_path, out, tables = sys.argv[1], sys.argv[2], sys.argv[3:]
key = ["alphabet", "ksize", "setting"]
frames = [pl.read_csv(t, separator="\t", infer_schema_length=None) for t in tables]
df = (
    pl.concat(frames, how="diagonal_relaxed")
    if frames
    else pl.DataFrame(schema={"alphabet": pl.Utf8, "ksize": pl.Int64, "setting": pl.Utf8})
)
expected = pl.read_csv(expected_path, separator="\t", schema_overrides={"penalty": pl.Utf8})
failed = expected.join(df.select(key).unique(), on=key, how="anti").with_columns(
    pl.lit("failed").alias("status")
)
df = pl.concat([df, failed], how="diagonal_relaxed")
df = df.sort([c for c in key + ["score", "level"] if c in df.columns], nulls_last=True)
df.write_csv(out, separator="\t")
print(df.unique(key).group_by("status").len().sort("status"), file=sys.stderr)
