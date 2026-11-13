"""Check 6b: (1) how many verdict exons each tool's calls COVER (overlap >= 0.8 of the exon,
no limit on call length), to show the structure tools do align across the short exons;
(2) landing counts per single kmerseek setting."""

import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import polars as pl
from common import *

v = verdict_exons().with_row_index("eid")
for tool in [
    "kmerseek",
    "foldseek",
    "reseek",
    "prostt5",
    "hmmer3_phmmer",
    "mmseqs2_seqseq",
    "hhblits",
]:
    c = load_calls(tool)
    j = v.select(
        "eid", "query_acc", "species", "target_acc", "prot_start", "prot_end"
    ).join(
        c.select("query_acc", "species", "target_acc", "tstart", "tend"),
        on=["query_acc", "species", "target_acc"],
    )
    ov = pl.min_horizontal("tend", "prot_end") - pl.max_horizontal(
        "tstart", "prot_start"
    )
    cov = j.filter(ov >= 0.8 * (pl.col("prot_end") - pl.col("prot_start")))[
        "eid"
    ].n_unique()
    print(
        f"{tool:16s} exons covered >=80% by some call: {cov:5d} / {v.height} ({100*cov/v.height:.1f}%)"
    )
km = load_calls("kmerseek")
lr = landing_rows(v, km)
per = (
    lr.group_by("setting")
    .agg(pl.col("eid").n_unique().alias("n"))
    .sort("n", descending=True)
)
print("best single kmerseek settings (setting, exons landed):", per.head(5).rows())
print(
    "median single setting:",
    per["n"].median(),
    " settings landing on 0:",
    404 - per.height,
)
