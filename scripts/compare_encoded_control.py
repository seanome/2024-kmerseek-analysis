#!/usr/bin/env python3
"""Positive control for notebook 246: does an encoded partition run as protein20 give the
same regions as kmerseek's own alphabet with that partition?

Compares two kmerseek region tables, the built-in one (--reference, e.g.
human_vs_zebrafish.hp_pbotc_1st_ed2.k19.lcfalse.regions.parquet) and the encoded one
(--encoded, e.g. human_vs_zebrafish.pbotc_encoded2.k19.lcfalse.regions.parquet). Reports

  rows in each table
  (query, target, region_start, region_end) in only one of the two tables. One query
    region can match several places on a target, so this is a set, not a row count
  rows, keyed by those four plus target_start and target_end (unique in a region table),
    in only one table
  rows in both whose region_n_shared_kmers or region_length differ
  rows in both whose region_enrichment, region_tail_probability or
    region_expected_shared_kmers differ by more than 1e-9 relative (enrichment and the
    tail probability are what notebook 244's scoring uses)

and exits 1 when the differing rows exceed --max-diff-frac (0.1%) of the larger table.
Writes the counts as one JSON line to stdout, and --out if given.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import polars as pl

REGION = ["query_name", "target_name", "region_start", "region_end"]
KEY = REGION + ["target_start", "target_end"]
EXACT = ["region_n_shared_kmers", "region_length"]
SCORES = [
    "region_enrichment",
    "region_tail_probability",
    "region_expected_shared_kmers",
]


def compare(reference: Path, encoded: Path, rel_tol: float = 1e-9) -> dict:
    cols = KEY + EXACT + SCORES
    ref = pl.scan_parquet(reference).select(cols)
    enc = pl.scan_parquet(encoded).select(cols)
    n_ref = ref.select(pl.len()).collect().item()
    n_enc = enc.select(pl.len()).collect().item()
    ref_set = ref.select(REGION).unique()
    enc_set = enc.select(REGION).unique()
    n_regions_ref = ref_set.select(pl.len()).collect().item()
    n_regions_enc = enc_set.select(pl.len()).collect().item()
    regions_only_ref = (
        ref_set.join(enc_set, on=REGION, how="anti").select(pl.len()).collect().item()
    )
    regions_only_enc = (
        enc_set.join(ref_set, on=REGION, how="anti").select(pl.len()).collect().item()
    )
    only_ref = ref.join(enc, on=KEY, how="anti").select(pl.len()).collect().item()
    only_enc = enc.join(ref, on=KEY, how="anti").select(pl.len()).collect().item()
    both = ref.join(enc, on=KEY, how="inner", suffix="_enc")
    exact_diff = pl.any_horizontal([pl.col(c) != pl.col(f"{c}_enc") for c in EXACT])
    score_diff = pl.any_horizontal(
        [
            (pl.col(c) - pl.col(f"{c}_enc")).abs()
            > rel_tol * pl.max_horizontal(pl.col(c).abs(), pl.lit(1e-300))
            for c in SCORES
        ]
    )
    counts = (
        both.select(
            n_both=pl.len(),
            n_exact_diff=exact_diff.sum(),
            n_score_diff=score_diff.sum(),
            n_any_diff=(exact_diff | score_diff).sum(),
        )
        .collect()
        .row(0, named=True)
    )
    n_differ = only_ref + only_enc + counts["n_any_diff"]
    return {
        "reference": str(reference),
        "encoded": str(encoded),
        "n_rows_reference": n_ref,
        "n_rows_encoded": n_enc,
        "n_regions_reference": n_regions_ref,
        "n_regions_encoded": n_regions_enc,
        "n_regions_only_reference": regions_only_ref,
        "n_regions_only_encoded": regions_only_enc,
        "n_only_reference": only_ref,
        "n_only_encoded": only_enc,
        **counts,
        "n_rows_differing": n_differ,
        "frac_differing": n_differ / max(n_ref, n_enc, 1),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--encoded", type=Path, required=True)
    ap.add_argument("--max-diff-frac", type=float, default=0.001)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    res = compare(args.reference, args.encoded)
    res["max_diff_frac"] = args.max_diff_frac
    res["pass"] = res["frac_differing"] <= args.max_diff_frac
    line = json.dumps(res)
    print(line)
    if args.out:
        args.out.write_text(line + "\n")
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
