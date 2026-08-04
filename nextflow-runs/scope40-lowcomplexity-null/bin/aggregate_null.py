#!/usr/bin/env python3
"""
Concatenate per-task complexity tables into the four files the notebook reads.

  hits.parquet             every hit from every (seqset, alphabet, ksize), with
                           its minority-count histogram
  kmers_by_minority.parquet  matched-k-mer complexity distribution
  freq_by_minority.parquet   reference k-mer frequency vs minority count
  run_stats.tsv            per-task counts, including how many hits were
                           dropped as non-contiguous and the encoding check

Usage:
    aggregate_null.py --outdir .
"""

from __future__ import annotations

import argparse
import glob
import sys

import polars as pl


def concat_parquet(pattern: str, out_path: str) -> int:
    files = sorted(glob.glob(pattern))
    frames = []
    for path in files:
        frame = pl.read_parquet(path)
        if not frame.is_empty():
            frames.append(frame)
    if not frames:
        # Still write the file so downstream stages have something to open.
        pl.DataFrame().write_parquet(out_path)
        print(f"warning: no non-empty inputs for {pattern}", file=sys.stderr)
        return 0
    combined = pl.concat(frames, how="diagonal_relaxed")
    combined.write_parquet(out_path)
    print(f"{out_path}: {len(combined):,} rows from {len(frames)} files", file=sys.stderr)
    return len(combined)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default=".")
    args = parser.parse_args()

    concat_parquet("*.hits.parquet", f"{args.outdir}/hits.parquet")
    concat_parquet("*.kmers.parquet", f"{args.outdir}/kmers_by_minority.parquet")
    concat_parquet("*.freq_by_minority.parquet", f"{args.outdir}/freq_by_minority.parquet")
    concat_parquet("*.top_kmers.parquet", f"{args.outdir}/top_kmers.parquet")

    stat_files = sorted(glob.glob("*.stats.tsv")) + sorted(glob.glob("*.freq_stats.tsv"))
    frames = [pl.read_csv(f, separator="\t") for f in stat_files]
    if frames:
        stats = pl.concat(frames, how="diagonal_relaxed")
        stats.write_csv(f"{args.outdir}/run_stats.tsv", separator="\t")
        print(f"run_stats.tsv: {len(stats):,} rows", file=sys.stderr)

        # Surface the invariants rather than burying them in a file nobody opens.
        if "encoding_check_mismatches" in stats.columns:
            bad = stats.filter(pl.col("encoding_check_mismatches") > 0)
            if not bad.is_empty():
                print("FATAL: HP encoding check failed for some tasks", file=sys.stderr)
                print(bad, file=sys.stderr)
                return 1
        if "n_hits_noncontiguous_dropped" in stats.columns:
            dropped = stats["n_hits_noncontiguous_dropped"].fill_null(0).sum()
            total = stats["n_hits_raw"].fill_null(0).sum()
            if total:
                print(
                    f"non-contiguous hits dropped: {dropped:,} / {total:,} "
                    f"({100 * dropped / total:.2f}%)",
                    file=sys.stderr,
                )

    shuffle_reports = sorted(glob.glob("*.shuffle_report.tsv"))
    if shuffle_reports:
        rows = []
        for path in shuffle_reports:
            row = {}
            for line in open(path):
                key, _, value = line.rstrip("\n").partition("\t")
                row[key] = value
            rows.append(row)
        pl.DataFrame(rows).write_csv(f"{args.outdir}/shuffle_reports.tsv", separator="\t")
        print(f"shuffle_reports.tsv: {len(rows)} shuffles", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
