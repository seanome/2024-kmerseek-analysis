#!/usr/bin/env python3
"""Summarise the SCOPe40 combination test: eight searches combined, on real and shuffled queries.

Inputs (copied from Sherlock, sha256 checked), written by bin/combine_alphabets.py:
  ~/data/scope/results-scope40-alphabet-ksize-v040-combine/scope40_v040_combined_{auc,bands}.tsv
      head job 47932901, 2026-10-15, real queries
  ~/data/scope/results-scope40-alphabet-ksize-v040-combine-shuffled-k2-seed1/...
      head job 47932904, 2026-10-15, dipeptide-shuffled queries, seed 1
The eight searches: hsdm17 k5, mmseqs12 k6, dayhoff6 k8, sdm12 k7, protein20 k5, uniprot18 k5
and wwmj5 k9 at c_max, hp_thomas_dill2 k18 at C = 2.

Writes:
  tables/277_scope40_combined_auc.tsv    per method and level: AUC over FoldSeek's 5_713
                                         queries on both scales, real and shuffled queries
  tables/277_scope40_combined_bands.tsv  per method, pair class and identity band: pairs ranked
                                         above the first false positive, real and shuffled
"""

import hashlib
import sys
from pathlib import Path

import polars as pl

REPO = Path(__file__).resolve().parents[1]
DATA = Path.home() / "data/scope"
REAL = DATA / "results-scope40-alphabet-ksize-v040-combine"
SHUF = DATA / "results-scope40-alphabet-ksize-v040-combine-shuffled-k2-seed1"
SHA256 = {
    REAL / "scope40_v040_combined_auc.tsv": "81065533380b849f03df81d7cf0b0f1744ca14093f938ea26931b9254753b3a4",
    REAL / "scope40_v040_combined_bands.tsv": "b1e6bdfec2e2e65594444aef5ee793ea4c15f96a6b167e64169ffa6aa94188be",
    SHUF / "scope40_v040_combined_auc.tsv": "1c8d558cff2d0ad239d61241c10b7fb4ee17d8728fe6ea9c233567377404194c",
    SHUF / "scope40_v040_combined_bands.tsv": "f80458b38b3d025db390a417aee235b4c4944a1832e190c605017f065b805306",
}
for path, want in SHA256.items():
    got = hashlib.sha256(path.read_bytes()).hexdigest()
    if got != want:
        sys.exit(f"{path} has sha256 {got}, not {want}")


def read(path):
    return pl.read_csv(path, separator="\t", infer_schema_length=None).filter(pl.col("status") == "ok")


key = ["method", "level"]
auc = (read(REAL / "scope40_v040_combined_auc.tsv").select(key + ["auc_ref_db", "auc_ref"])
       .join(read(SHUF / "scope40_v040_combined_auc.tsv")
             .select(key + [pl.col("auc_ref_db").alias("auc_ref_db_shuffled"),
                            pl.col("auc_ref").alias("auc_ref_shuffled")]), on=key)
       .sort("level", "auc_ref_db", descending=[False, True]))
auc.write_csv(REPO / "tables/277_scope40_combined_auc.tsv", separator="\t")

key = ["method", "category", "identity_band"]
bands = (read(REAL / "scope40_v040_combined_bands.tsv")
         .select(key + ["n_pairs", pl.col("n_reported").alias("reported"),
                        pl.col("n_above_first_fp").alias("above_first_fp")])
         .join(read(SHUF / "scope40_v040_combined_bands.tsv")
               .select(key + [pl.col("n_reported").alias("reported_shuffled"),
                              pl.col("n_above_first_fp").alias("above_first_fp_shuffled")]), on=key)
         .sort(key))
bands.write_csv(REPO / "tables/277_scope40_combined_bands.tsv", separator="\t")

with pl.Config(tbl_rows=60, tbl_width_chars=200, float_precision=5):
    print("Family and superfamily AUC, database scale (FoldSeek's), real and shuffled queries")
    print(auc.filter(pl.col("level").is_in(["family", "superfamily"]))
          .pivot(on="level", index="method", values=["auc_ref_db", "auc_ref_db_shuffled"])
          .sort("auc_ref_db_family", descending=True))
    print("Under 20% identity: pairs above the first false positive, real (shuffled)")
    print(bands.filter(pl.col("identity_band") == "0-20%")
          .with_columns(pl.format("{} ({})", "above_first_fp", "above_first_fp_shuffled").alias("real_shuffled"))
          .pivot(on="category", index="method", values="real_shuffled"))
