"""Check 6: what do the non-kmerseek calls look like? Are they whole-protein alignments
that the 80%-of-call-inside-exon rule can never accept?"""

import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import polars as pl
from common import *

m = models()
tlen = m.filter(pl.col("side") == "target").select(
    "species", pl.col("accession").alias("target_acc"), "n_res"
)
v = verdict_exons()
vpairs = v.select("species", "target_acc", "query_acc").unique()
fl = exons().group_by("species", "target_acc").agg(pl.col("target_len").first())
for tool in ["kmerseek", "foldseek", "reseek", "prostt5", "hmmer3_phmmer", "hhblits"]:
    c = load_calls(tool).join(fl, on=["species", "target_acc"], how="left")
    c = c.with_columns((pl.col("tend") - pl.col("tstart")).alias("L"))
    cv = c.join(vpairs, on=["species", "target_acc", "query_acc"], how="inner")
    npairs = cv.select("species", "target_acc").unique().height
    print(
        f"{tool:15s} calls on verdict pairs={cv.height:8d} verdict pairs with any call={npairs:3d}/{vpairs.height} "
        f"call len median={cv['L'].median()} p10={cv['L'].quantile(0.1)} "
        f"frac calls <=75 res={(cv['L']<=75).mean():.3f} "
        f"median call/target len={(cv['L']/cv['target_len']).median():.2f}"
    )
    # out-of-range coordinates
    bad = c.filter(
        (pl.col("tstart") < 0)
        | (pl.col("tend") > pl.col("target_len"))
        | (pl.col("tend") <= pl.col("tstart"))
    )
    print(f"   calls with coords outside [0,target_len] or empty: {bad.height}")
# How many exons could a tool land on at all: it needs a call of length <= exon_len/0.8 = 75
