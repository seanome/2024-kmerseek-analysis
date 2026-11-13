#!/usr/bin/env python3
"""Positive control for notebook 246: does an encoded partition run as protein20 give the
same regions as kmerseek's own alphabet with that partition?

Compares two kmerseek region tables, the built-in one (--reference, e.g.
human_vs_zebrafish.hp_pbotc_1st_ed2.k19.lcfalse.extend-c2.regions.parquet) and the encoded
one (--encoded, e.g. human_vs_zebrafish.encoded_hp_pbotc_1st_ed2.k19.lcfalse.extend-c2...). Reports

  rows in each table
  (query, target, region_start, region_end) in only one table. One query region can match
    several places on a target, so this is a set, not a row count
  rows, keyed by those four plus target_start and target_end (unique in a region table),
    in only one table
  gated: rows in both whose region_n_shared_kmers or region_length differ, or whose
    region_mean_idf, region_enrichment or region_tail_probability differ by more than 1e-9
    relative. Enrichment and the tail probability are what notebook 244's Bonferroni cut
    uses; all of these come from the index's k-mer counts, which encoding does not change
  reported, not gated: region_evalue and region_ka_bits. Their Karlin-Altschul fit is made
    on shuffled target sequences; kmerseek shuffles amino acids for a built-in alphabet and
    A/D for the encoded one, so the two fits differ a little by construction. Reported as
    the number of rows differing, the largest ratio between the two E-values, and the rows
    that pass --evalue-cut in one table and not the other

Exits 1 when the gated differing rows exceed --max-diff-frac (0.1%) of the larger table.
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
GATED_SCORES = ["region_mean_idf", "region_enrichment", "region_tail_probability"]
REPORTED_SCORES = ["region_evalue", "region_ka_bits"]


def _differs(c: str, rel_tol: float) -> pl.Expr:
    a, b = pl.col(c), pl.col(f"{c}_enc")
    scale = pl.max_horizontal(a.abs(), b.abs(), pl.lit(1e-300))
    both_null = a.is_null() & b.is_null()
    return ~both_null & (a.is_null() | b.is_null() | ((a - b).abs() > rel_tol * scale))


def compare(
    reference: Path, encoded: Path, rel_tol: float = 1e-9, evalue_cut: float = 0.01
) -> dict:
    names = set(pl.scan_parquet(reference).collect_schema().names()) & set(
        pl.scan_parquet(encoded).collect_schema().names()
    )
    gated = [c for c in GATED_SCORES if c in names]
    reported = [c for c in REPORTED_SCORES if c in names]
    cols = KEY + EXACT + gated + reported
    ref = pl.scan_parquet(reference).select(cols)
    enc = pl.scan_parquet(encoded).select(cols)
    n_ref = ref.select(pl.len()).collect().item()
    n_enc = enc.select(pl.len()).collect().item()
    ref_set = ref.select(REGION).unique()
    enc_set = enc.select(REGION).unique()
    res = {
        "reference": str(reference),
        "encoded": str(encoded),
        "n_rows_reference": n_ref,
        "n_rows_encoded": n_enc,
        "n_regions_reference": ref_set.select(pl.len()).collect().item(),
        "n_regions_encoded": enc_set.select(pl.len()).collect().item(),
        "n_regions_only_reference": ref_set.join(enc_set, on=REGION, how="anti")
        .select(pl.len())
        .collect()
        .item(),
        "n_regions_only_encoded": enc_set.join(ref_set, on=REGION, how="anti")
        .select(pl.len())
        .collect()
        .item(),
        "n_only_reference": ref.join(enc, on=KEY, how="anti")
        .select(pl.len())
        .collect()
        .item(),
        "n_only_encoded": enc.join(ref, on=KEY, how="anti")
        .select(pl.len())
        .collect()
        .item(),
        "gated_columns": EXACT + gated,
        "reported_columns": reported,
    }
    both = ref.join(enc, on=KEY, how="inner", suffix="_enc")
    gate = pl.any_horizontal(
        [pl.col(c) != pl.col(f"{c}_enc") for c in EXACT]
        + [_differs(c, rel_tol) for c in gated]
    )
    aggs = [pl.len().alias("n_both"), gate.sum().alias("n_gated_diff")]
    for c in reported:
        aggs.append(_differs(c, rel_tol).sum().alias(f"n_{c}_diff"))
    if "region_evalue" in reported:
        e, f = pl.col("region_evalue"), pl.col("region_evalue_enc")
        ratio = pl.max_horizontal(e, f) / pl.min_horizontal(e, f).clip(
            lower_bound=1e-300
        )
        aggs += [
            ratio.max().alias("max_evalue_ratio"),
            ratio.median().alias("median_evalue_ratio"),
            ((e <= evalue_cut) != (f <= evalue_cut))
            .sum()
            .alias(f"n_pass_evalue_{evalue_cut}_in_one_only"),
            (e <= evalue_cut).sum().alias(f"n_pass_evalue_{evalue_cut}_reference"),
            (f <= evalue_cut).sum().alias(f"n_pass_evalue_{evalue_cut}_encoded"),
        ]
    res.update(both.select(aggs).collect().row(0, named=True))
    n_differ = res["n_only_reference"] + res["n_only_encoded"] + res["n_gated_diff"]
    res["n_rows_differing_gated"] = n_differ
    res["frac_differing_gated"] = n_differ / max(n_ref, n_enc, 1)
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--encoded", type=Path, required=True)
    ap.add_argument("--max-diff-frac", type=float, default=0.001)
    ap.add_argument("--evalue-cut", type=float, default=0.01)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    res = compare(args.reference, args.encoded, evalue_cut=args.evalue_cut)
    res["max_diff_frac"] = args.max_diff_frac
    res["pass"] = res["frac_differing_gated"] <= args.max_diff_frac
    line = json.dumps(res)
    print(line)
    if args.out:
        args.out.write_text(line + "\n")
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
