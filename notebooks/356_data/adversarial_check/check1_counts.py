"""Check 1: verdict-exon count and landing counts per tool."""

import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import polars as pl
from common import *

v = verdict_exons().with_row_index("eid")
print(
    "verdict exons:",
    v.height,
    " by species:",
    v.group_by("species").len().sort("species").rows(),
)
print(
    "unique (species,target,start,end):",
    v.unique(["species", "target_acc", "prot_start", "prot_end"]).height,
)
print("pairs represented:", v.select("species", "target_acc").unique().height)
e = exons()
pr = pairs()
e0 = e.filter(pl.col("internal") & (pl.col("exon_len") <= 60)).join(
    pr, on=["species", "target_acc"]
)
print("internal<=60 on pairs, before model filter:", e0.height)
print("exon_len range in verdict:", v["exon_len"].min(), v["exon_len"].max())
res = {}
for tool in [
    "kmerseek",
    "foldseek",
    "reseek",
    "prostt5",
    "hmmer3_phmmer",
    "hmmer3_jackhmmer",
    "mmseqs2_seqseq",
    "mmseqs2_iterative",
    "hhblits",
]:
    c = load_calls(tool)
    s = lands(v, c)
    res[tool] = s
    print(
        f"{tool:20s} calls={c.height:9d} settings={c['setting'].n_unique():4d} "
        f"lands={len(s):5d} ({100*len(s)/v.height:.1f}%)"
    )
v.with_columns(
    pl.col("eid").is_in(list(res["kmerseek"])).alias("km_land")
).write_parquet(f"{HERE}/my_verdict_exons.parquet")
