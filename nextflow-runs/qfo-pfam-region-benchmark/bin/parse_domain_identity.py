#!/usr/bin/env python3
"""Reduce MMseqs2 domain-vs-domain hits to one identity per human domain instance.

Record ids are <accession>|<pfam_id>|<start>-<end>, written by extract_domain_sequences.py.
"""

import argparse
from pathlib import Path

import polars as pl

SCHEMA = {
    "accession": pl.String, "pfam_id": pl.String, "domain_start": pl.Int64,
    "domain_end": pl.Int64, "best_pident": pl.Float64, "best_alnlen": pl.Int64,
    # WHICH target won. Kept so a covariate of the target -- its disorder, its length --
    # can be attached to the human instance the same way best_pident already is.
    "best_target": pl.String,
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hits", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    if args.hits.stat().st_size == 0:
        pl.DataFrame(schema=SCHEMA).write_parquet(args.out, compression="zstd")
        print("no hits; wrote empty identity table")
        return

    lf = pl.scan_csv(args.hits, separator="\t", has_header=False,
                     new_columns=["q", "t", "pident", "alnlen", "evalue"])
    q, t = pl.col("q").str.split("|"), pl.col("t").str.split("|")
    df = (
        lf.select(
            q.list.get(0).alias("accession"),
            q.list.get(1).alias("pfam_id"),
            q.list.get(2).alias("q_range"),
            t.list.get(0).alias("target_accession"),
            t.list.get(1).alias("t_pfam"),
            pl.col("pident").cast(pl.Float64),
            pl.col("alnlen").cast(pl.Int64),
        )
        # Same family only. A cross-family hit is not the homolog whose identity is wanted,
        # and would understate difficulty by reporting some unrelated closer sequence.
        .filter(pl.col("pfam_id") == pl.col("t_pfam"))
        .collect(engine="streaming")
    )
    if df.height == 0:
        pl.DataFrame(schema=SCHEMA).write_parquet(args.out, compression="zstd")
        print("no same-family hits; wrote empty identity table")
        return

    rng = pl.col("q_range").str.split("-")
    df = df.with_columns(
        rng.list.get(0).cast(pl.Int64).alias("domain_start"),
        rng.list.get(1).cast(pl.Int64).alias("domain_end"),
    )
    # Best available homolog per instance: the EASIEST case a tool could have transferred
    # from, so the identity bin is an upper bound on how hard that instance was.
    #
    # One ROW is chosen, not three independent maxima. This used to be
    #   .agg(pl.col("pident").max(), pl.col("alnlen").max())
    # which takes each column's maximum separately, so best_alnlen was the longest
    # alignment of ANY same-family hit rather than the alignment length of the hit that
    # produced best_pident. The two came from different targets whenever the closest
    # homolog was not also the longest alignment, which is the common case. Nothing could
    # have been attributed to a target at all under that shape.
    #
    # Sorted before grouping and taking first() so the winning row is a real row.
    # alnlen then target_accession break ties on pident, which keeps the choice
    # deterministic across runs rather than dependent on input order.
    best = (
        df.sort(["pident", "alnlen", "target_accession"],
                descending=[True, True, False], nulls_last=True)
        .group_by("accession", "pfam_id", "domain_start", "domain_end", maintain_order=True)
        .agg(pl.col("pident").first().alias("best_pident"),
             pl.col("alnlen").first().alias("best_alnlen"),
             pl.col("target_accession").first().alias("best_target"))
    )
    best.write_parquet(args.out, compression="zstd")
    print(f"{best.height} human domain instances with a same-family target match")
    print(best["best_pident"].describe())


if __name__ == "__main__":
    main()
