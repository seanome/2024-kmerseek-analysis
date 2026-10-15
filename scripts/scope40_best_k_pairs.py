#!/usr/bin/env python3
"""Print the alphabet:k pairs the SCOPe40 identity-band run scores first.

From the first sweep's table (scope40_v040_auc.tsv, Sherlock run 2026-10-10 to 10-12, sha256
c9a30b83d6c0a064b79f629c0fe921d6366d3b14b9d801cb06101077319101af), as in notebook 277: hits
ranked by region_evalue (extension settings) or jaccard (exact), AUC over FoldSeek's 5_713
queries. For every alphabet it keeps the k with the best family AUC and the k with the best
superfamily AUC. For the hydrophobic-polar (HP) alphabets, the ones the identity question is
about, it also keeps every k from k_min to k_min + HP_EXTRA.

The output is the value of BEST_K_PAIRS in nextflow-runs/scope40-alphabet-ksize-v040/Makefile.

Run:  python scripts/scope40_best_k_pairs.py [path/to/scope40_v040_auc.tsv]
"""

import hashlib
import sys
from pathlib import Path

import polars as pl

SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else (
    Path.home() / "data/scope/results-scope40-alphabet-ksize-v040/scope40_v040_auc.tsv")
SRC_SHA256 = "c9a30b83d6c0a064b79f629c0fe921d6366d3b14b9d801cb06101077319101af"
HP_EXTRA = 2  # every HP peak was at k_min, k_min + 1 or k_min + 2 (notebook 277, section 4)

digest = hashlib.sha256(SRC.read_bytes()).hexdigest()
if digest != SRC_SHA256:
    sys.exit(f"{SRC} has sha256 {digest}, not {SRC_SHA256}")

auc = pl.read_csv(SRC, separator="\t")
ok = auc.filter(pl.col("status") == "ok", pl.col("level").is_in(["family", "superfamily"]))
chosen = ok.filter(
    ((pl.col("setting") != "exact") & (pl.col("score") == "region_evalue"))
    | ((pl.col("setting") == "exact") & (pl.col("score") == "jaccard"))
)
best = chosen.sort("auc_ref", descending=True).group_by("alphabet", "level").head(1)
k_min = auc.group_by("alphabet").agg(pl.col("ksize").min().alias("k_min"))
hp = k_min.filter(pl.col("alphabet").str.starts_with("hp_")).select(
    "alphabet", pl.int_ranges("k_min", pl.col("k_min") + HP_EXTRA + 1).alias("ksize")).explode("ksize")
pairs = (pl.concat([best.select("alphabet", "ksize"), hp.with_columns(pl.col("ksize").cast(pl.Int64))])
         .unique().sort("alphabet", "ksize"))
print(f"# {pairs.height} pairs", file=sys.stderr)
print(",".join(f"{a}:{k}" for a, k in pairs.iter_rows()))
