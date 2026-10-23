"""Notebook 287: do notebook 284's searches of the real queries give the same E-values as
notebook 241's, setting by setting?

284 searched Ced9, P66 and BHF again (with their shuffled copies) on notebook 283's rebuilt
indexes. While it ran, 9 of those indexes gained new data files (a fit stored at search
time) and another 283 job was searching them. If the fitted K had changed, the shuffled
copies would be scored on a different scale from the real queries in sections 2 and 3.

Input: the real queries' calls with a finite E-value from 284's output, extracted on
Sherlock to REAL_284 (2026-10-20), and 241's regions.parquet.
Output: notebooks/287_check_284_matches_241.csv, one row per setting.
"""

from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
D = Path("/Users/olga/data/botryllus/alphabet-ranking-three-cases")
REAL_284 = D / "287/real_queries_in_284.parquet"
KEYS = [
    "query_name",
    "target_name",
    "region_start",
    "region_end",
    "target_start",
    "setting",
]


def setting() -> pl.Expr:
    return pl.col("alphabet") + ".k" + pl.col("ksize_arm").cast(pl.String)


def main() -> None:
    a = (
        pl.read_parquet(REAL_284)
        .with_columns(pl.col("query_name").str.strip_chars(), setting=setting())
        .select(*KEYS, e284="region_evalue")
    )
    b = (
        pl.scan_parquet(D / "regions.parquet")
        .filter(pl.col("region_evalue").is_finite())
        .with_columns(setting=setting())
        .select(*KEYS, e241="region_evalue")
        .collect()
    )
    j = a.join(b, on=KEYS, how="full", coalesce=True).with_columns(
        ratio=pl.col("e284") / pl.col("e241")
    )
    out = (
        j.group_by("setting")
        .agg(
            n_both=(pl.col("e284").is_not_null() & pl.col("e241").is_not_null()).sum(),
            n_only_284=pl.col("e241").is_null().sum(),
            n_only_241=pl.col("e284").is_null().sum(),
            ratio_min=pl.col("ratio").min(),
            ratio_max=pl.col("ratio").max(),
        )
        .sort("setting")
    )
    out.write_csv(ROOT / "notebooks/287_check_284_matches_241.csv")
    off = out.filter((pl.col("ratio_min") < 0.9999) | (pl.col("ratio_max") > 1.0001))
    print(
        f"{out['n_both'].sum()} calls in both, {out['n_only_284'].sum()} only in 284, "
        f"{out['n_only_241'].sum()} only in 241; {off.height} settings with an E-value ratio off 1"
    )
    print(out.filter((pl.col("n_only_284") > 0) | (pl.col("n_only_241") > 0)))


if __name__ == "__main__":
    main()
