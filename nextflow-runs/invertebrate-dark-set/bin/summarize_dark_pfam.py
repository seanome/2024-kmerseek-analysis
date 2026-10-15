#!/usr/bin/env python3
"""Turn the Pfam scan of the dark set into a table and a summary.

What the number is for: the dark set's headline is "no sequence arm placed these", and the
first question asked of it is whether they are real proteins at all. compare_dark_lengths.py
answers that with length, which is a proxy. This answers it directly -- a dark protein
carrying a recognisable Pfam domain is a real protein that phmmer, jackhmmer and mmseqs2
still failed to place.

That is not circular. Darkness is defined by three pairwise SEQUENCE searches against
reviewed Swiss-Prot with the query's clade removed. Pfam is a library of profile HMMs built
from curated alignments, which is a different and more sensitive instrument and a different
database. A protein can carry a Pfam domain and be dark, and that subset is the point.

A domain counts when it passes Pfam's own gathering threshold (hmmsearch --cut_ga), which
is the curated per-family score Pfam uses to decide membership. --i-evalue can tighten that
further but defaults to leaving it alone.

The counts are reported over the dark set as the denominator, taken from the dark parquet
rather than from the rows present here, so a protein with no Pfam hit still counts in the
denominator instead of vanishing.
"""
import argparse
import json
import sys
from pathlib import Path

import polars as pl

# hmmsearch --domtblout, 1-indexed as the awk in the module emits them:
#   1 sequence, 4 HMM name, 5 HMM accession, 20/21 env coords, 14 domain bitscore,
#   13 i-Evalue. hmmsearch, not hmmscan: the scan direction iterates models over one
#   sequence at a time and is the slow way round when there are many sequences.
COLUMNS = ["accession", "pfam_name", "pfam_acc", "env_start", "env_end",
           "bitscore", "i_evalue"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hits", type=Path, required=True, help="the tsv.gz from hmmsearch")
    ap.add_argument("--dark", type=Path, required=True, help="<species>_dark_set.parquet")
    ap.add_argument("--species", required=True)
    # The real threshold is hmmsearch --cut_ga, Pfam's own curated per-family gathering
    # score, which is what "this protein has this domain" means in Pfam. This knob only
    # tightens further and defaults to off; setting it below 1 discards calls Pfam's
    # curators accepted, so it is a deliberate act and not a default.
    ap.add_argument("--i-evalue", type=float, default=1.0,
                    help="optional extra independent-E-value ceiling on top of --cut_ga "
                         "(default 1.0, which keeps everything --cut_ga accepted)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--summary-out", type=Path, required=True)
    args = ap.parse_args()

    dark = pl.read_parquet(args.dark).select("accession").unique()
    n_dark = dark.height

    try:
        hits = pl.read_csv(args.hits, separator="\t", has_header=False,
                           new_columns=COLUMNS, infer_schema_length=0)
    except Exception:
        # No hits at all is a real answer. An empty gzip stream is not a zero-byte file,
        # so this catches the empty case by failing to parse rather than by a size check.
        hits = pl.DataFrame(schema={c: pl.Utf8 for c in COLUMNS})

    if hits.height:
        hits = hits.with_columns(
            pl.col("env_start").cast(pl.Int64, strict=False),
            pl.col("env_end").cast(pl.Int64, strict=False),
            pl.col("bitscore").cast(pl.Float64, strict=False),
            pl.col("i_evalue").cast(pl.Float64, strict=False),
            # Pfam accessions are versioned in the HMM library (PF00001.24); every other
            # table in this project keys on the unversioned id.
            pl.col("pfam_acc").str.split(".").list.get(0).alias("pfam_acc"),
        ).filter(pl.col("i_evalue") <= args.i_evalue)

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

    print(f"{args.species}: {n_with} of {n_dark} dark proteins carry a Pfam domain "
          f"(--cut_ga, i-Evalue <= {args.i_evalue})", file=sys.stderr)
    if dropped:
        print(f"WARNING: {dropped} hit rows name a protein not in the dark set",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
