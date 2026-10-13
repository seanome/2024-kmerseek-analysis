#!/usr/bin/env python3
"""Compare the SCOPe40 identity bands of real and shuffled queries, search for search.

Inputs (copied from Sherlock, sha256 checked):
  ~/data/scope/results-scope40-alphabet-ksize-v040-best-k-identity/scope40_v040_identity_bands.tsv
      step 2 of the PR 107 follow-up (head job 47604583, 2026-10-12): real queries
  ~/data/scope/results-scope40-alphabet-ksize-v040-shuffled-k2-seed1/scope40_v040_identity_bands.tsv
      the same 148 searches with every query dipeptide-shuffled, seed 1 (head jobs 47646678
      and 47665023, 2026-10-13)

A shuffled query keeps its SCOP labels, so every pair it ranks above the query's first false
positive (different fold; for class pairs, different class) is chance. Hits ranked by
region_evalue (extension settings) or jaccard (exact), as in notebook 277.

Writes tables/277_scope40_shuffled_vs_real_identity_bands.tsv: per search, pair class and
identity band, the pairs reported and ranked above the first false positive with real and with
shuffled queries.
"""

import hashlib
import sys
from pathlib import Path

import polars as pl

REPO = Path(__file__).resolve().parents[1]
DATA = Path.home() / "data/scope"
REAL = DATA / "results-scope40-alphabet-ksize-v040-best-k-identity/scope40_v040_identity_bands.tsv"
SHUF = DATA / "results-scope40-alphabet-ksize-v040-shuffled-k2-seed1/scope40_v040_identity_bands.tsv"
SHA256 = {
    REAL: "e5b2f81fed537bff806319f07758352dda185cae3e72e897fd573616d29e1ae6",
    SHUF: "049683dee204b953ec8c6326f4936737c160986f514685215b10924dd9ed9860",
}
for path, want in SHA256.items():
    got = hashlib.sha256(path.read_bytes()).hexdigest()
    if got != want:
        sys.exit(f"{path} has sha256 {got}, not {want}")

KEY = ["alphabet", "ksize", "setting", "category", "identity_band"]


def ranked(path):
    b = pl.read_csv(path, separator="\t", infer_schema_length=None).filter(pl.col("status") == "ok")
    return b.filter(((pl.col("setting") != "exact") & (pl.col("score") == "region_evalue"))
                    | ((pl.col("setting") == "exact") & (pl.col("score") == "jaccard")))


real = ranked(REAL).select(KEY + ["n_pairs", pl.col("n_reported").alias("reported_real"),
                                  pl.col("n_above_first_fp").alias("above_real")])
shuf = ranked(SHUF).select(KEY + [pl.col("n_reported").alias("reported_shuffled"),
                                  pl.col("n_above_first_fp").alias("above_shuffled")])
both = real.join(shuf, on=KEY, how="full", coalesce=True).sort(KEY)
missing = both.filter(pl.col("above_real").is_null() | pl.col("above_shuffled").is_null())
if missing.height:
    sys.exit(f"{missing.height} rows are in one run only:\n{missing}")
both.write_csv(REPO / "tables/277_scope40_shuffled_vs_real_identity_bands.tsv", separator="\t")
print(both.select(["alphabet", "ksize", "setting"]).unique().height, "searches in both runs")

under20 = both.filter(pl.col("identity_band") == "0-20%")
print("Under 20% identity, summed over all searches")
print(under20.group_by("category").agg(pl.col("above_real").sum(), pl.col("above_shuffled").sum(),
                                       pl.col("reported_real").sum(), pl.col("reported_shuffled").sum())
      .sort("category"))
print("Under 20% identity, each alphabet's search with the most real pairs above the first false positive")
with pl.Config(tbl_rows=80, tbl_width_chars=200):
    print(under20.sort("above_real", descending=True).group_by("alphabet", "category").head(1)
          .sort("category", "above_real", descending=[False, True])
          .select("category", "alphabet", "ksize", "setting", "n_pairs", "above_real", "above_shuffled"))
