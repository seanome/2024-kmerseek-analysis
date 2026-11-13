"""Compare my numbers with the notebook's saved outputs (run after my own checks)."""

import sys, os, json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import polars as pl
from common import *

T = f"{REPO}/tables/356"
vj = json.load(open(f"{T}/verdict.json"))
mine = pl.read_parquet(f"{HERE}/my_best_pplace.parquet")
print(
    "my exons with a landing call of p_place < 0.05:",
    mine.filter(pl.col("best_p") < 0.05).height,
    " notebook:",
    vj["rule_3"]["kmerseek_lands_with_p_place_lt_0.05"][0],
)
nv = pl.read_csv(f"{T}/null_kmerseek_variants.csv")
print("notebook null draws:", nv.height)
for col, obs in [
    ("best_of_all", vj["rule_1"]["kmerseek_best_of_all_lands"][2]),
    (
        "best_of_all_p_place_lt_0.05",
        vj["rule_3"]["kmerseek_lands_with_p_place_lt_0.05"][2],
    ),
    ("single_sdm12_k6_lcTrue", vj["kmerseek_best_single_setting"]["lands"][2]),
]:
    s = nv[col]
    print(
        f"{col:30s} observed={obs:.4f} null median={s.median():.4f} max={s.max():.4f} "
        f"draws >= observed: {(s >= obs).sum()} of {len(s)}  excess exons vs mean={1669*(obs - s.mean()):.1f}"
    )
et = pl.read_parquet(f"{T}/exon_tool.parquet")
ev = et.filter(pl.col("verdict_set"))
print("notebook verdict exons:", ev.select("species", "exon_id").unique().height)
nk = (
    ev.filter(pl.col("tool") == "kmerseek_best_of_all")
    .group_by("species", "exon_id")
    .agg(
        pl.col("lands").any().alias("nb_land"),
        pl.col("p_place").filter(pl.col("lands")).min().alias("nb_p"),
    )
)
j = mine.join(nk, on=["species", "exon_id"], how="full", coalesce=True)
print(
    "exons only in mine / only in notebook:",
    j["eid"].null_count(),
    j["nb_land"].null_count(),
)
myland = j["best_p"].is_not_null()
print("landing disagreements:", (myland != j["nb_land"].fill_null(False)).sum())
both = j.filter(pl.col("best_p").is_not_null() & pl.col("nb_p").is_not_null())
print("best p_place max abs diff:", (both["best_p"] - both["nb_p"]).abs().max())
for t in ["foldseek", "reseek", "prostt5"]:
    print(
        t,
        "notebook lands:",
        ev.filter(pl.col("tool") == t)["lands"].sum(),
        " rows:",
        ev.filter(pl.col("tool") == t).height,
    )
