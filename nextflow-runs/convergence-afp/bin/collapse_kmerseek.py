#!/usr/bin/env python3
"""Reduce one kmerseek search to what the convergence test reads.

kmerseek reports every region it finds; the test ranks targets. Two files come out:

  <label>.best.parquet    one row per query and target: the smallest region_evalue of
                          any region (the ranking score), the largest region_ka_bits and
                          the number of regions. Every pair kmerseek reported.
  <label>.truth.parquet   every region, all columns, on a target in the truth set: what
                          the residue blocks of the hits are drawn from.

The CSV is read lazily from disk, uncompressed, because polars inflates a zstd CSV in
memory before parsing it.
"""

import argparse
import sys
from pathlib import Path

import polars as pl


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--regions", type=Path, required=True, help="kmerseek search CSV")
    ap.add_argument("--truth", type=Path, required=True, help="truth_set.tsv")
    ap.add_argument("--label", required=True)
    args = ap.parse_args()

    truth = pl.read_csv(args.truth, separator="\t")["Entry"].to_list()
    best_out, truth_out = Path(f"{args.label}.best.parquet"), Path(
        f"{args.label}.truth.parquet"
    )
    if args.regions.stat().st_size == 0:
        sys.exit(
            f"{args.regions} is empty: kmerseek wrote no header, so this is not a no-hit result"
        )
    lf = pl.scan_csv(args.regions, infer_schema_length=10_000)
    cols = lf.collect_schema().names()
    need = {"query_name", "target_name", "region_evalue", "region_ka_bits"}
    if not need <= set(cols):
        sys.exit(f"columns missing from {args.regions}: {sorted(need - set(cols))}")
    (
        lf.group_by("query_name", "target_name")
        .agg(
            pl.col("region_evalue").cast(pl.Float64).min().alias("min_region_evalue"),
            pl.col("region_ka_bits").cast(pl.Float64).max().alias("max_region_ka_bits"),
            pl.len().alias("n_regions"),
        )
        .with_columns(setting_label=pl.lit(args.label))
        .sink_parquet(best_out, compression="zstd")
    )
    (
        lf.filter(pl.col("target_name").is_in(truth))
        .with_columns(setting_label=pl.lit(args.label))
        .sink_parquet(truth_out, compression="zstd")
    )
    b = pl.read_parquet(best_out)
    print(
        f"{args.label}: {b.height} query-target pairs, "
        f"{b['query_name'].n_unique()} queries with a hit, "
        f"{pl.read_parquet(truth_out).height} regions on truth targets"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
