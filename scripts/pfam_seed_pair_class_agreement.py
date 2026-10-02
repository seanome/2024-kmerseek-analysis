#!/usr/bin/env python3
"""Per-pair class agreement, kappa and longest exact run for Pfam seed pairs, every alphabet.

Reads the Pfam-A 38.2 seed pair alignments written by `scripts/pfam_seed_pair_alignments.py`
and writes one row per (pair, alphabet) for every alphabet in
`notebooks/hp_conservation_utils.ALPHABET_CLUSTERS` (all 19 kmerseek alphabets). Notebooks
230 and 234 read the output.

Usage:
    python3 scripts/pfam_seed_pair_class_agreement.py [--check-against OLD.parquet]

With `--check-against OLD`, rows for alphabets already in OLD are kept from OLD and only the
other alphabets are added. Before that, every deterministic column (n_cols, agree, expected,
kappa, longest_run) of the recomputed rows must equal OLD exactly (NaN kappa, from a pair whose
residues all fall in one class, counts as equal), and the mean of `longest_run_null` per
alphabet must agree within NULL_MEAN_TOL. The null is a random shuffle, and the table of
2026-09-13 was made with a different random draw than this script's seed 0, so row-level null
values differ; keeping OLD's rows means no number already reported from it changes.
"""

import argparse
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks"))
import hp_conservation_utils as hc  # noqa: E402

PFAM_DIR = Path("/Users/olga/data/pfam")
ALIGNMENTS = PFAM_DIR / "230_pfam_seed_pair_alignments.parquet"
OUT = PFAM_DIR / "230_pfam_seed_pair_class_agreement.parquet"
KEY_COLS = ["family", "family_id", "n_seed", "query", "target", "q_len", "t_len", "lali", "seqid_ali"]
EXACT_COLS = ["n_cols", "agree", "expected", "kappa", "longest_run"]
NULL_MEAN_TOL = 0.02  # relative difference allowed in the per-alphabet mean shuffle-null run


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--alignments", type=Path, default=ALIGNMENTS)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--check-against", type=Path, default=None)
    args = ap.parse_args()

    aln = pl.read_parquet(args.alignments)
    table = hc.compute_pair_table(aln, KEY_COLS)
    print(f"{aln.height:_} pairs x {table['alphabet'].n_unique()} alphabets = {table.height:_} rows")

    if args.check_against:
        old = pl.read_parquet(args.check_against)
        joined = old.join(table, on=["query", "target", "alphabet"], how="left", suffix="_new")
        n_missing = joined.filter(pl.col("n_cols_new").is_null()).height
        bad = {
            c: joined.filter(
                ~((pl.col(c) == pl.col(f"{c}_new")) | (pl.col(c).is_nan() & pl.col(f"{c}_new").is_nan()))
            ).height
            for c in EXACT_COLS
        }
        null_means = joined.group_by("alphabet").agg(
            pl.col("longest_run_null").mean().alias("old"), pl.col("longest_run_null_new").mean().alias("new")
        ).with_columns(((pl.col("new") - pl.col("old")).abs() / pl.col("old")).alias("rel_diff"))
        print(f"checked {old.height:_} old rows: {n_missing} missing, differing per column: {bad}")
        print(null_means.sort("alphabet"))
        if n_missing or any(bad.values()) or null_means["rel_diff"].max() > NULL_MEAN_TOL:
            sys.exit("new table does not reproduce the old one; not written")
        added = table.filter(~pl.col("alphabet").is_in(old["alphabet"].unique().to_list()))
        print(f"keeping the old rows, adding {added['alphabet'].unique().sort().to_list()}")
        table = pl.concat([old, added.select(old.columns)])

    table.write_parquet(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
