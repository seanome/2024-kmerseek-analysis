#!/usr/bin/env python3
"""Turn the Pfam scan of the dark set into a table and a summary.

A Pfam profile match supports protein-like sequence in a dark-set entry. It does not
validate the complete gene model or independently establish evolutionary homology.
Darkness here means no qualifying match in the specified conventional searches against
a clade-excluded reference. Pfam uses another sequence-derived reference and may share
training data with those searches; it is supporting annotation, not independent truth.

A domain counts when it passes Pfam's own gathering threshold (hmmsearch --cut_ga), which
is the curated per-family score Pfam uses to decide membership. --i-evalue can tighten that
further but defaults to leaving it alone.

The counts are reported over the dark set as the denominator, taken from the dark parquet
rather than from the rows present here, so a protein with no Pfam hit still counts in the
denominator instead of vanishing.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import polars as pl

# hmmsearch --domtblout, 1-indexed as the awk in the module emits them:
#   1 sequence, 4 HMM name, 5 HMM accession, 20/21 env coords, 14 domain bitscore,
#   13 i-Evalue. hmmsearch, not hmmscan: the scan direction iterates models over one
#   sequence at a time and is the slow way round when there are many sequences.
COLUMNS = [
    "accession",
    "pfam_name",
    "pfam_acc",
    "env_start",
    "env_end",
    "bitscore",
    "i_evalue",
]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--hits", type=Path, required=True, help="the tsv.gz from hmmsearch"
    )
    ap.add_argument(
        "--dark", type=Path, required=True, help="<species>_dark_set.parquet"
    )
    ap.add_argument("--species", required=True)
    ap.add_argument(
        "--i-evalue",
        type=float,
        default=None,
        help="optional finite, nonnegative independent-E-value ceiling; "
        "default: keep all calls passing --cut_ga",
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--summary-out", type=Path, required=True)
    args = ap.parse_args()

    if args.i_evalue is not None and (
        not math.isfinite(args.i_evalue) or args.i_evalue < 0
    ):
        ap.error("--i-evalue must be finite and nonnegative")

    dark = pl.read_parquet(args.dark).select("accession").unique()
    n_dark = dark.height

    try:
        hits = pl.read_csv(
            args.hits,
            separator="\t",
            has_header=False,
            new_columns=COLUMNS,
            infer_schema_length=0,
        )
    except pl.exceptions.NoDataError:
        # No hits at all is a real answer. An empty gzip stream is not a zero-byte file,
        # so this catches the empty case by failing to parse rather than by a size check.
        hits = pl.DataFrame(
            schema={
                c: (
                    pl.Int64
                    if c in {"env_start", "env_end"}
                    else pl.Float64 if c in {"bitscore", "i_evalue"} else pl.Utf8
                )
                for c in COLUMNS
            }
        )

    if hits.width != len(COLUMNS):
        raise ValueError(f"Expected {len(COLUMNS)} Pfam columns; got {hits.width}")

    if hits.height:
        hits = hits.with_columns(
            pl.col("env_start").cast(pl.Int64),
            pl.col("env_end").cast(pl.Int64),
            pl.col("bitscore").cast(pl.Float64),
            pl.col("i_evalue").cast(pl.Float64),
            # Pfam accessions are versioned in the HMM library (PF00001.24); every other
            # table in this project keys on the unversioned id.
            pl.col("pfam_acc").str.split(".").list.get(0).alias("pfam_acc"),
        )
        invalid = hits.filter(
            pl.any_horizontal([pl.col(c).is_null() for c in COLUMNS])
            | ~pl.col("pfam_acc").str.contains(r"^PF[0-9]{5}$")
            | (pl.col("env_start") < 1)
            | (pl.col("env_end") < pl.col("env_start"))
            | ~pl.col("bitscore").is_finite()
            | ~pl.col("i_evalue").is_finite()
            | (pl.col("i_evalue") < 0)
        )
        if invalid.height:
            raise ValueError(f"Invalid Pfam rows: {invalid.height}")
        unknown = hits.join(dark, on="accession", how="anti")
        if unknown.height:
            raise ValueError(f"Pfam rows outside the dark query set: {unknown.height}")
        if args.i_evalue is not None:
            hits = hits.filter(pl.col("i_evalue") <= args.i_evalue)

    # Only rows whose protein is actually in the dark set. The scan runs on the dark FASTA
    # so this should be every row; an inner join that drops any is a mismatch worth seeing
    # rather than a silent trim, so the count is reported.
    kept = hits.join(dark, on="accession", how="inner") if hits.height else hits
    dropped = hits.height - kept.height

    kept.write_parquet(args.out, compression="zstd")

    n_with = kept["accession"].n_unique() if kept.height else 0
    summary = {
        "species": args.species,
        "call_rule": "hmmsearch --cut_ga",
        "i_evalue_threshold": args.i_evalue,
        "proteins_dark": n_dark,
        "dark_proteins_with_pfam": n_with,
        "dark_proteins_without_pfam": n_dark - n_with,
        "fraction_of_dark_with_pfam": round(n_with / n_dark, 6) if n_dark else None,
        "pfam_domain_instances": kept.height,
        "distinct_pfam_families": kept["pfam_acc"].n_unique() if kept.height else 0,
        "hit_rows_not_in_dark_set": dropped,
    }
    args.summary_out.write_text(json.dumps(summary, indent=1))

    rule = (
        "--cut_ga"
        if args.i_evalue is None
        else f"--cut_ga, i-Evalue <= {args.i_evalue}"
    )
    print(
        f"{args.species}: {n_with} of {n_dark} dark proteins carry a Pfam domain "
        f"({rule})",
        file=sys.stderr,
    )
    if dropped:
        print(
            f"WARNING: {dropped} hit rows name a protein not in the dark set",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
