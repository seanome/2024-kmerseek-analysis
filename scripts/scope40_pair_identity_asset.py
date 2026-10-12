#!/usr/bin/env python3
"""Write the SCOPe40 pair identity table the kmerseek 0.4 sweep scores by identity band.

Input: notebook 230's USalign alignments of sampled SCOPe40 domain pairs
(`scripts/scope_pair_structural_alignments.py`, ~/data/scope/230_scope40_pair_alignments.parquet,
sha256 4d99389f11c4dc5340d94cb4320e2c8c0f746f47ffbb623a3b37e633214f6ef7). 38_267 pairs in four
classes: same family, same superfamily but different family, same fold but different
superfamily, and different fold in the same class. `seqid_ali` is the share of identical
residues over the columns USalign aligned by structure, so it does not depend on a sequence
alignment.

The sampled pairs are unordered. kmerseek searches in one direction, so each pair is written
twice, once with each domain as the query. The different-fold pairs are left out: the
pipeline already treats every different-fold hit as a false positive.

Output: nextflow-runs/scope40-alphabet-ksize-v040/assets/scope40_pair_identity.tsv.gz
  query_domain, target_domain, category, seqid_ali

Run:  python scripts/scope40_pair_identity_asset.py [path/to/230_scope40_pair_alignments.parquet]
"""

import hashlib
import sys
from pathlib import Path

import polars as pl

REPO = Path(__file__).resolve().parents[1]
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "data/scope/230_scope40_pair_alignments.parquet"
SRC_SHA256 = "4d99389f11c4dc5340d94cb4320e2c8c0f746f47ffbb623a3b37e633214f6ef7"
OUT = REPO / "nextflow-runs/scope40-alphabet-ksize-v040/assets/scope40_pair_identity.tsv.gz"
KEEP = ["same_family", "same_superfamily_diff_family", "same_fold_diff_superfamily"]

digest = hashlib.sha256(SRC.read_bytes()).hexdigest()
if digest != SRC_SHA256:
    sys.exit(f"{SRC} has sha256 {digest}, not {SRC_SHA256}")

pairs = pl.read_parquet(SRC).filter(pl.col("category").is_in(KEEP), pl.col("seqid_ali").is_not_null())
one = pairs.select(pl.col("query").alias("query_domain"), pl.col("target").alias("target_domain"),
                   "category", "seqid_ali")
other = one.select(pl.col("target_domain").alias("query_domain"), pl.col("query_domain").alias("target_domain"),
                   "category", "seqid_ali")
both = pl.concat([one, other]).unique(["query_domain", "target_domain"]).sort("query_domain", "target_domain")
OUT.write_bytes(__import__("gzip").compress(both.write_csv(separator="\t").encode(), mtime=0))
print(OUT, both.height, "directed pairs from", pairs.height, "pairs")
print(both.group_by("category").len().sort("category"))
