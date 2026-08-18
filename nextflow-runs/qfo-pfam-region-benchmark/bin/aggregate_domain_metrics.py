#!/usr/bin/env python3
"""Concatenate every per-(tool, variant, species) metrics row and PR/ROC curve.

Emits parquet for notebooks and CSV for reading at the terminal, plus a leaderboard so
the sweep's headline is visible without opening anything.
"""

import sys
from pathlib import Path

import polars as pl

LEAD = ["tool", "variant", "species"]

# Threshold-free and therefore comparable across tools with different default cutoffs.
HEADLINE = ["roc_auc", "auprc", "best_f1", "recall_reachable", "precision", "median_iou_tp"]


def concat(dirpath: Path, what: str) -> pl.DataFrame:
    files = sorted(dirpath.glob("*.parquet"))
    if not files:
        raise SystemExit(f"no {what} parquet files under {dirpath}")
    df = pl.concat([pl.read_parquet(f) for f in files], how="diagonal_relaxed")
    return df.select(LEAD + [c for c in df.columns if c not in LEAD])


def main():
    metrics_dir, curves_dir = Path(sys.argv[1]), Path(sys.argv[2])
    parquet_out, csv_out, curves_out = (Path(a) for a in sys.argv[3:6])

    metrics = concat(metrics_dir, "metrics").sort(
        ["auprc", "best_f1"], descending=True, nulls_last=True
    )
    metrics.write_parquet(parquet_out, compression="zstd")
    metrics.write_csv(csv_out)

    curves = concat(curves_dir, "curve")
    curves.write_parquet(curves_out, compression="zstd")

    print(f"{metrics.height} metric rows, {curves.height} curve points")
    print()

    # Best combo per tool. kmerseek has 113 variants and every other tool has one, so a
    # flat sort would bury the baselines under the sweep.
    best = (
        metrics.sort("auprc", descending=True, nulls_last=True)
        .group_by("tool")
        .agg([pl.col("variant").first().alias("best_variant")]
             + [pl.col(c).first() for c in HEADLINE])
        .sort("auprc", descending=True, nulls_last=True)
    )
    with pl.Config(tbl_cols=-1, tbl_rows=-1, fmt_str_lengths=40):
        print(best)


if __name__ == "__main__":
    main()
