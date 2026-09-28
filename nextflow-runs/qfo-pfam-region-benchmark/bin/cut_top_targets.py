#!/usr/bin/env python3
"""Keep each query's best N target proteins in a kmerseek regions parquet.

A target's score is its best region's --rank-by value (higher is better). For each query,
every region of every target whose score is at least the query's N-th best target score is
kept, so targets tied at the cut all stay. That makes the cut safe to apply before any
scoring step that cuts to the same N with its own tie-break: that step picks from the same
targets it would have seen in the uncut table. Queries with N or fewer targets keep them all.

Why: kmerseekSearch keeps every match (--max-query-pvalue 1.0 in the ELM cover search), and
on the chicken dry run that was 717 GB of regions for 272 arms, while the cover score reads
only each query's top 1_000 targets. `--kmerseek_top_targets N` runs this inside the search
task, so a run over more target species writes what is read and nothing else.

    cut_top_targets.py --in regions.parquet --out cut.parquet --top 1000 [--rank-by region_mean_idf]
"""

import argparse
import sys

import polars as pl


def cut(lf: pl.LazyFrame, top: int, rank_by: str) -> pl.LazyFrame:
    best = lf.group_by("query_name", "target_name").agg(pl.col(rank_by).max().alias("_best"))
    floor = (best.group_by("query_name")
             .agg(pl.col("_best").sort(descending=True).head(top).min().alias("_floor")))
    keep = (best.join(floor, on="query_name")
            .filter(pl.col("_best") >= pl.col("_floor"))
            .select("query_name", "target_name"))
    return lf.join(keep, on=["query_name", "target_name"], how="semi")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--top", type=int, required=True)
    ap.add_argument("--rank-by", default="region_mean_idf")
    args = ap.parse_args()
    if args.top < 1:
        sys.exit("--top must be at least 1")

    lf = pl.scan_parquet(args.inp)
    if args.rank_by not in lf.collect_schema().names():
        sys.exit(f"{args.inp} has no `{args.rank_by}` column to rank targets by")
    n_in = lf.select(pl.len()).collect().item()
    cut(lf, args.top, args.rank_by).sink_parquet(args.out, compression="zstd", compression_level=9)
    n_out = pl.scan_parquet(args.out).select(pl.len()).collect().item()
    print(f"kept {n_out:,} of {n_in:,} regions: each query's best {args.top} targets by {args.rank_by}")


if __name__ == "__main__":
    main()
