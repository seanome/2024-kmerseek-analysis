#!/usr/bin/env python3
"""Compare the SCOPe40 searches below k_min (step 1 of the PR 107 follow-up) with the first sweep.

Inputs (copied from Sherlock, sha256 checked):
  ~/data/scope/results-scope40-alphabet-ksize-v040/scope40_v040_auc.tsv
      first sweep, 2026-10-10 to 10-12, every alphabet at k_min..k_max
  ~/data/scope/results-scope40-alphabet-ksize-v040-below-kmin/scope40_v040_auc.tsv
  ~/data/scope/results-scope40-alphabet-ksize-v040-below-kmin/scope40_v040_identity_bands.tsv
      step 1, 2026-10-12 (head job 47546661), k_min-1 and k_min-2 for 16 alphabets

Hits ranked by region_evalue (extension settings) or jaccard (exact), as in notebook 277.
AUC over FoldSeek's 5_713 queries (auc_ref), a tie counted against kmerseek.

Writes:
  tables/277_scope40_below_kmin_vs_first_sweep.tsv   best AUC per alphabet and level, both runs
  tables/277_scope40_below_kmin_identity_0_20.tsv    share of under-20%-identity pairs ranked
                                                     above the query's first different-fold hit
"""

import hashlib
import sys
from pathlib import Path

import polars as pl

REPO = Path(__file__).resolve().parents[1]
DATA = Path.home() / "data/scope"
FIRST = DATA / "results-scope40-alphabet-ksize-v040/scope40_v040_auc.tsv"
BELOW = DATA / "results-scope40-alphabet-ksize-v040-below-kmin/scope40_v040_auc.tsv"
BANDS = DATA / "results-scope40-alphabet-ksize-v040-below-kmin/scope40_v040_identity_bands.tsv"
SHA256 = {
    FIRST: "c9a30b83d6c0a064b79f629c0fe921d6366d3b14b9d801cb06101077319101af",
    BELOW: "de5aa346313e36a30e50850c67b466b91bb418398a02676d3e0c9061299e3ea6",
    BANDS: "5eaf0d189e02b674a0eca7a38750e172864c007f8ca215202a352cca0a90deaf",
}
for path, want in SHA256.items():
    got = hashlib.sha256(path.read_bytes()).hexdigest()
    if got != want:
        sys.exit(f"{path} has sha256 {got}, not {want}")


def ranked(d):
    """Rows of the score column notebook 277 ranks by: region_evalue, or jaccard for exact."""
    return d.filter(pl.col("status") == "ok").filter(
        ((pl.col("setting") != "exact") & (pl.col("score") == "region_evalue"))
        | ((pl.col("setting") == "exact") & (pl.col("score") == "jaccard")))


def best(d, tag):
    return (ranked(d).sort("auc_ref", descending=True).group_by("alphabet", "level").head(1)
            .select("alphabet", "level", pl.col("ksize").alias(f"k_{tag}"),
                    pl.col("setting").alias(f"setting_{tag}"), pl.col("auc_ref").alias(f"auc_{tag}")))


first = pl.read_csv(FIRST, separator="\t")
below = pl.read_csv(BELOW, separator="\t")
k_min = first.group_by("alphabet").agg(pl.col("ksize").min().alias("k_min"))
cmp = (best(below, "below").join(best(first, "first"), on=["alphabet", "level"]).join(k_min, on="alphabet")
       .filter(pl.col("level").is_in(["family", "superfamily", "fold"]))
       .with_columns((pl.col("auc_below") - pl.col("auc_first")).alias("change"))
       .sort("level", "auc_below", descending=[False, True]))
cmp.write_csv(REPO / "tables/277_scope40_below_kmin_vs_first_sweep.tsv", separator="\t")
print(cmp.group_by("level").agg((pl.col("change") > 0).sum().alias("n_better_below_kmin"),
                                pl.len().alias("n_alphabets"), pl.col("change").max().alias("largest_gain")))

bands = ranked(pl.read_csv(BANDS, separator="\t", infer_schema_length=None)).filter(
    pl.col("identity_band") == "0-20%")
share = (bands.with_columns((pl.col("n_above_first_fp") / pl.col("n_pairs")).alias("share_above_first_fp"))
         .sort("share_above_first_fp", descending=True).group_by("alphabet", "category").head(1)
         .select("alphabet", "category", "ksize", "setting", "n_pairs", "n_reported", "n_above_first_fp",
                 "share_above_first_fp")
         .sort("category", "share_above_first_fp", descending=[False, True]))
share.write_csv(REPO / "tables/277_scope40_below_kmin_identity_0_20.tsv", separator="\t")
print(share.group_by("category").agg(pl.col("share_above_first_fp").max(), pl.col("n_pairs").first()))
