#!/usr/bin/env python3
"""
format_kmerseek_results.py

Turn kmerseek's region table (one row per matched region of a query-target pair) into
the 4-column pair table the evaluator reads: the best score per (query, target).

Usage:
    format_kmerseek_results.py <regions.csv> <out.tsv.gz>
        [--rank-by max_containment] [--max-bonferroni-p 0.05]

Output TSV (gzipped, no header):
    query_name  target_name  score  corrected_p

`--rank-by` names the column whose per-pair MAXIMUM becomes the score. The default,
max_containment, is a whole-pair statistic and what this benchmark has always ranked on.
region_enrichment (the region benchmark's default) is per region; its maximum over the
pair's regions is the pair score.

`--max-bonferroni-p` drops regions whose Poisson tail, corrected for how many positions
the region could start at (region_search_space) and how many targets were searched
(db_n_targets), is not below the threshold -- the recipe kmerseek's own source gives and
the region benchmark applies. 0 disables it. A pair whose every region is dropped is
absent from the output, which the evaluator scores as 0.

An empty region table (kmerseek found nothing, or the CSV is header-only) writes an
empty output; that is a real result, not an error.
"""

import argparse
import gzip
import sys
from pathlib import Path

import polars as pl


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("regions_csv", type=Path)
    p.add_argument("out_tsv_gz", type=Path)
    p.add_argument("--rank-by", default="max_containment")
    p.add_argument("--max-bonferroni-p", type=float, default=0.05)
    args = p.parse_args()

    if args.regions_csv.stat().st_size == 0:
        _write_empty(args.out_tsv_gz, "empty region table")
        return

    try:
        lf = pl.scan_csv(args.regions_csv, ignore_errors=True)
        names = lf.collect_schema().names()
    except pl.exceptions.NoDataError:
        _write_empty(args.out_tsv_gz, "header-only region table")
        return

    for col in ("query_name", "target_name", args.rank_by):
        if col not in names:
            sys.exit(
                f"{args.regions_csv} has no `{col}` column. Columns present: {sorted(names)}. "
                f"A file written by an older kmerseek may predate the field."
            )

    corrected = pl.lit(1.0)
    if args.max_bonferroni_p > 0:
        needed = {"region_search_space", "db_n_targets"}
        has_p = "region_tail_probability" in names or "region_poisson_score" in names
        if not needed.issubset(names) or not has_p:
            missing = sorted((needed | {"region_tail_probability"}) - set(names))
            sys.exit(
                f"{args.regions_csv} is missing {missing}, so the Bonferroni filter cannot be "
                f"applied. Pass --max-bonferroni-p 0 to disable it."
            )
        raw_p = (
            pl.col("region_tail_probability").cast(pl.Float64)
            if "region_tail_probability" in names
            else 10.0 ** -pl.col("region_poisson_score").cast(pl.Float64)
        )
        n_tests = pl.col("region_search_space").cast(pl.Float64) * pl.col("db_n_targets").cast(pl.Float64)
        corrected = pl.min_horizontal(raw_p * n_tests, pl.lit(1.0))
        lf = lf.with_columns(corrected.alias("_corrected_p")).filter(pl.col("_corrected_p") < args.max_bonferroni_p)
    else:
        lf = lf.with_columns(corrected.alias("_corrected_p"))

    pairs = (
        lf.select(
            pl.col("query_name").cast(pl.String),
            pl.col("target_name").cast(pl.String),
            pl.col(args.rank_by).cast(pl.Float64).alias("score"),
            pl.col("_corrected_p"),
        )
        .drop_nulls(["query_name", "target_name", "score"])
        .group_by(["query_name", "target_name"])
        .agg(pl.col("score").max(), pl.col("_corrected_p").min())
        .sort(["query_name", "target_name"])
        .collect()
    )

    with gzip.open(args.out_tsv_gz, "wt") as out:
        for q, t, s, cp in pairs.iter_rows():
            out.write(f"{q}\t{t}\t{s}\t{cp}\n")
    print(f"Wrote {len(pairs)} pairs to {args.out_tsv_gz} (rank_by={args.rank_by}, "
          f"bonferroni<{args.max_bonferroni_p})", file=sys.stderr)


def _write_empty(path: Path, why: str) -> None:
    with gzip.open(path, "wt"):
        pass
    print(f"{why}: wrote 0 pairs to {path}", file=sys.stderr)


if __name__ == "__main__":
    main()
