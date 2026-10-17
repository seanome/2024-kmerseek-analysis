#!/usr/bin/env python3
"""Keep each query's best N target proteins in a kmerseek regions parquet, by one or more scores.

Each --rank-by is COLUMN:max (higher is better) or COLUMN:min (lower is better). A target's
score under a ranking is its best region's value. For each query and each ranking, the
targets whose score is at least as good as the query's N-th best are kept, so targets tied
at the cut all stay. A target stays if any ranking keeps it: with two rankings a query keeps
between N and 2N targets. Values that are not finite are left out of a ranking; an arm whose
Karlin-Altschul fit was refused has region_evalue = inf on every row, and counting those
would tie every target at the cut and keep them all. That makes the cut safe to apply before any
scoring step that cuts to the same N with its own tie-break: that step picks from the same
targets it would have seen in the uncut table. Queries with N or fewer targets keep them all.

Why: kmerseekSearch keeps every match (--max-query-pvalue 1.0 in the ELM cover search), and
on the chicken dry run that was 717 GB of regions for 272 arms, while the cover score reads
only each query's top 1_000 targets. `--kmerseek_top_targets N` runs this inside the search
task, so a run over more target species writes what is read and nothing else.

    cut_top_targets.py --in regions.parquet --out cut.parquet --top 1000 \
        --rank-by region_mean_idf:max --rank-by region_evalue:min

The ELM cover search keeps the union of region_mean_idf and region_evalue: the cover score
ranks by region_mean_idf, which on the chicken run followed call length (median Spearman 0.80
over 266 arms), and a cut by that score alone could never be re-ranked by the E-value later.
"""

import argparse
import sys

import polars as pl


# Queries and (query, target) pairs are grouped by 64-bit hashes of their names, not by the
# names: FASTA headers made the grouping cost 11.9 GB on a 30-million-row chicken arm. Two
# pairs sharing a hash would need a 64-bit collision among a few million pairs.
QUERY = pl.col("query_name").hash(seed=0)
PAIR = pl.struct("query_name", "target_name").hash(seed=0)


def kept_by(lf: pl.LazyFrame, top: int, column: str, direction: str) -> pl.LazyFrame:
    """Hashes of the (query, target) pairs in each query's top `top` targets under one ranking."""
    higher = direction == "max"
    v = pl.col(column).cast(pl.Float64)
    best = (lf.filter(v.is_finite())
            .select(_q=QUERY, _p=PAIR, _v=v)
            .group_by("_q", "_p")
            .agg((pl.col("_v").max() if higher else pl.col("_v").min()).alias("_best")))
    edge = (best.group_by("_q")
            .agg(pl.col("_best").sort(descending=higher).head(top).last().alias("_edge")))
    at_least_as_good = pl.col("_best") >= pl.col("_edge") if higher else pl.col("_best") <= pl.col("_edge")
    return best.join(edge, on="_q").filter(at_least_as_good).select("_p")


def cut(lf: pl.LazyFrame, top: int, rankings: list[tuple[str, str]]) -> pl.LazyFrame:
    # The pairs to keep are collected first, one 8-byte hash per pair. The regions then
    # stream through a join against that small table.
    keep = (pl.concat([kept_by(lf, top, c, d) for c, d in rankings]).unique()
            .collect(engine="streaming"))
    return (lf.with_columns(_p=PAIR).join(keep.lazy(), on="_p", how="semi").drop("_p"))


def parse_ranking(spec: str) -> tuple[str, str]:
    column, _, direction = spec.partition(":")
    direction = direction or "max"
    if direction not in ("max", "min"):
        raise SystemExit(f"--rank-by {spec}: the direction after ':' is max or min")
    return column, direction


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--top", type=int, required=True)
    ap.add_argument("--rank-by", action="append",
                    help="COLUMN:max or COLUMN:min, repeatable (default region_mean_idf:max)")
    args = ap.parse_args()
    if args.top < 1:
        sys.exit("--top must be at least 1")

    rankings = [parse_ranking(r) for r in (args.rank_by or ["region_mean_idf:max"])]
    lf = pl.scan_parquet(args.inp)
    names = lf.collect_schema().names()
    missing = [c for c, _ in rankings if c not in names]
    if missing:
        sys.exit(f"{args.inp} has no {missing} column to rank targets by")
    n_in = lf.select(pl.len()).collect().item()
    cut(lf, args.top, rankings).sink_parquet(args.out, compression="zstd", compression_level=9,
                                             engine="streaming")
    n_out = pl.scan_parquet(args.out).select(pl.len()).collect().item()
    by = " or ".join(f"{c} ({'highest' if d == 'max' else 'lowest'})" for c, d in rankings)
    print(f"kept {n_out:,} of {n_in:,} regions: each query's best {args.top} targets by {by}")


if __name__ == "__main__":
    main()
