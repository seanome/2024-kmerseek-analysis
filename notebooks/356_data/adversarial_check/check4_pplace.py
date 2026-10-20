"""Check 4: placement p-value, closed form vs brute force over every start position,
for landing kmerseek calls. Also writes each exon's best (smallest) p_place for comparison.
"""

import sys, os, math, random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import polars as pl
from common import *


def lands1(s, L, a, b):
    ov = min(s + L, b) - max(s, a)
    return ov > 0 and ov >= 0.8 * L and ov >= 0.3 * (b - a)


def p_formula(L, N, a, b):
    E = b - a
    need = max(math.ceil(0.8 * L), math.ceil(0.3 * E))
    if need > min(L, E):
        return 0.0
    lo, hi = max(a + need - L, 0), min(b - need, N - L)
    return max(hi - lo + 1, 0) / (N - L + 1)


def p_brute(L, N, a, b):
    return sum(lands1(s, L, a, b) for s in range(N - L + 1)) / (N - L + 1)


v = verdict_exons().with_row_index("eid")
km = load_calls("kmerseek")
lr = landing_rows(v, km).join(v.select("eid", "target_len"), on="eid")
lr = lr.with_columns((pl.col("tend") - pl.col("tstart")).alias("L"))
print("landing (exon, call) rows:", lr.height)
random.seed(356)
idx = random.sample(range(lr.height), 10)
for i in idx:
    r = lr.row(i, named=True)
    f = p_formula(r["L"], r["target_len"], r["prot_start"], r["prot_end"])
    b = p_brute(r["L"], r["target_len"], r["prot_start"], r["prot_end"])
    print(
        f"{r['target_acc']:11s} {r['setting']:28s} call [{r['tstart']},{r['tend']}) L={r['L']:3d} exon [{r['prot_start']},{r['prot_end']}) "
        f"N={r['target_len']} formula={f:.5f} brute={b:.5f} {'OK' if abs(f-b)<1e-12 else 'DIFF'}"
    )
# wider test of the formula on 2,000 random landing rows
diff = 0
for i in random.sample(range(lr.height), 2000):
    r = lr.row(i, named=True)
    if (
        abs(
            p_formula(r["L"], r["target_len"], r["prot_start"], r["prot_end"])
            - p_brute(r["L"], r["target_len"], r["prot_start"], r["prot_end"])
        )
        > 1e-12
    ):
        diff += 1
print("formula vs brute force, 2000 random landing calls: differ on", diff)
# Note: the formula's ceil(0.8L) equals the brute-force rule because overlap is an integer.
best = (
    lr.with_columns(
        pl.struct("L", "target_len", "prot_start", "prot_end")
        .map_elements(
            lambda r: p_formula(
                r["L"], r["target_len"], r["prot_start"], r["prot_end"]
            ),
            return_dtype=pl.Float64,
        )
        .alias("p")
    )
    .group_by("eid")
    .agg(
        pl.col("p").min().alias("best_p"),
        pl.len().alias("n_landing_calls"),
        pl.col("setting").n_unique().alias("n_settings"),
    )
)
best = v.select(
    "eid", "species", "target_acc", "exon_id", "prot_start", "prot_end"
).join(best, on="eid", how="left")
best.write_parquet(f"{HERE}/my_best_pplace.parquet")
print(
    best.select(
        pl.col("best_p").median(),
        pl.col("n_landing_calls").median(),
        pl.col("n_settings").median(),
    )
)
print("distribution of number of settings that land, per landed exon:")
print(best.drop_nulls("n_settings")["n_settings"].describe())
