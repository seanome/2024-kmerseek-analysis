#!/usr/bin/env python3
"""Concatenate every per-(tool, variant, species) metrics row into one table.

Emits parquet for downstream notebooks and CSV for reading at the terminal without
starting python.
"""

import sys
from pathlib import Path

import polars as pl

LEAD = ["tool", "variant", "species"]
SORT = ["auprc", "f1_reachable"]


def main():
    metrics_dir, parquet_out, csv_out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])

    files = sorted(metrics_dir.glob("*.parquet"))
    if not files:
        raise SystemExit(f"no metrics parquet files under {metrics_dir}")

    df = pl.concat([pl.read_parquet(f) for f in files], how="diagonal_relaxed")
    df = df.select(LEAD + [c for c in df.columns if c not in LEAD])
    df = df.sort(SORT, descending=True, nulls_last=True)

    df.write_parquet(parquet_out, compression="zstd")
    df.write_csv(csv_out)

    print(f"{df.height} rows from {len(files)} files")
    # Best combo per tool -- the sweep's headline, without opening a notebook.
    best = df.group_by("tool").agg(
        pl.col("variant").first(),
        pl.col("auprc").max().alias("best_auprc"),
    ).sort("best_auprc", descending=True)
    print(best)


if __name__ == "__main__":
    main()
