#!/usr/bin/env python3
"""
Emit a table of (region_length - k + 1) for k in 6..30, against the
observed distribution of Pfam envelope lengths across all 10 species'
QfO annotations (results/pfam_benchmark/annotations/*_pfam_domains.parquet).

Purpose: once kmerseek search is region-scored (Prompt A) rather than
whole-protein-scored, the number of k-mers available in a region is
region_length - k + 1. Every k tested so far (18..30) was chosen for
whole-protein search, where the denominator is thousands of residues.
At domain scale, a k close to or above the region length yields 1, 0, or
a negative (degenerate/undefined) k-mer count. This table makes those
degenerate cells visible before any region-scored search runs, so the
eventual k-sweep (6..30) isn't wasting cells on combinations that can't
produce a signal.

Usage:
    python pfam_envelope_ksize_table.py
"""

from pathlib import Path

import polars as pl

ANNOT_DIR = Path("results/pfam_benchmark/annotations")
OUT_DIR = Path("results/pfam_benchmark/ksize_diagnostics")
K_MIN, K_MAX = 6, 30
QUANTILES = [0.0, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 1.0]


def load_envelope_lengths() -> pl.DataFrame:
    frames = []
    for f in sorted(ANNOT_DIR.glob("*_pfam_domains.parquet")):
        species = f.stem.replace("_pfam_domains", "")
        df = pl.read_parquet(f).filter(pl.col("has_position")).select([
            pl.lit(species).alias("species"),
            "pfam_id",
            (pl.col("domain_end") - pl.col("domain_start") + 1).alias("domain_length"),
        ])
        frames.append(df)
    return pl.concat(frames)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lengths = load_envelope_lengths()
    print(f"Envelope-annotated domain instances: {lengths.height:,} across {lengths['species'].n_unique()} species")

    lengths.write_parquet(OUT_DIR / "envelope_lengths.parquet", compression="snappy")

    q = lengths.select([
        pl.col("domain_length").quantile(p).alias(f"p{int(p*100):02d}") for p in QUANTILES
    ]).row(0, named=True)

    print("\n=== Observed Pfam envelope length distribution (all species) ===")
    for p in QUANTILES:
        key = f"p{int(p*100):02d}"
        print(f"  {key:>4s}: {q[key]:.0f} aa")

    # region_length - k + 1 table across quantile lengths and k=6..30
    rows = []
    for p in QUANTILES:
        key = f"p{int(p*100):02d}"
        region_length = q[key]
        for k in range(K_MIN, K_MAX + 1):
            n_kmers = region_length - k + 1
            rows.append({
                "quantile": key,
                "region_length": region_length,
                "k": k,
                "n_kmers_in_region": n_kmers,
                "degenerate": n_kmers <= 1,
            })
    table = pl.DataFrame(rows)
    table.write_parquet(OUT_DIR / "region_length_minus_k_plus_1.parquet", compression="snappy")

    print("\n=== region_length - k + 1, by envelope-length quantile x k (degenerate cells: n_kmers <= 1) ===")
    pivot = table.pivot(values="n_kmers_in_region", index="k", on="quantile", aggregate_function="first")
    pivot = pivot.sort("k")
    with pl.Config(tbl_rows=-1, tbl_cols=-1):
        print(pivot)

    n_degenerate = table.filter(pl.col("degenerate")).height
    n_total = table.height
    print(f"\nDegenerate (n_kmers<=1) cells: {n_degenerate}/{n_total} ({100*n_degenerate/n_total:.1f}%)")

    # smallest k that is non-degenerate at each quantile, and largest k still
    # non-degenerate at the low-length tail (p05/p10) -- the practical ceiling
    for key in ["p05", "p10", "p25", "p50"]:
        sub = table.filter(pl.col("quantile") == key).sort("k")
        ok = sub.filter(~pl.col("degenerate"))
        max_k_ok = ok["k"].max() if ok.height else None
        print(f"  at {key} envelope length ({q[key]:.0f} aa): largest non-degenerate k in 6..30 = {max_k_ok}")

    print(f"\nSaved: {OUT_DIR / 'envelope_lengths.parquet'}")
    print(f"Saved: {OUT_DIR / 'region_length_minus_k_plus_1.parquet'}")


if __name__ == "__main__":
    main()
