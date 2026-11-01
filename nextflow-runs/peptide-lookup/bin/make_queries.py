#!/usr/bin/env python3
"""Step 2 of notebook 300: one decoy per query peptide, and the query FASTA files.

Usage: make_queries.py <real_peptides.tsv> <spiked_peptides.tsv> <out_dir>

A decoy keeps the first and last residue and reverses everything between them, so it has the
same mass, length and residue counts as its peptide (and so the same class counts under any
alphabet), but its order is new. A decoy that comes out identical to its peptide (a
palindrome inside the ends) is kept and flagged, not dropped, so every peptide has one.

Writes queries.tsv (query_id, peptide_id, set real or spiked, is_decoy, sequence,
decoy_equals_target) and queries_targets.fasta, queries_decoys.fasta.
"""

import sys
from pathlib import Path

import polars as pl


def pseudo_reverse(seq):
    return seq[0] + seq[-2:0:-1] + seq[-1]


def main():
    real_tsv, spiked_tsv, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    real = (
        pl.read_csv(real_tsv, separator="\t")
        .select("peptide_id", sequence="Sequence")
        .with_columns(set=pl.lit("real"))
    )
    spiked = (
        pl.read_csv(spiked_tsv, separator="\t")
        .select("peptide_id", sequence="peptide")
        .with_columns(set=pl.lit("spiked"))
    )
    targets = pl.concat([real, spiked]).with_columns(is_decoy=pl.lit(False))
    decoys = targets.with_columns(
        sequence=pl.col("sequence").map_elements(pseudo_reverse, return_dtype=pl.Utf8),
        is_decoy=pl.lit(True),
    )
    queries = pl.concat([targets, decoys]).with_columns(
        query_id=pl.when(pl.col("is_decoy"))
        .then(pl.format("decoy_{}", "peptide_id"))
        .otherwise(pl.col("peptide_id"))
    )
    same = dict(zip(targets["peptide_id"], targets["sequence"]))
    queries = queries.with_columns(
        decoy_equals_target=pl.col("is_decoy")
        & (pl.col("sequence") == pl.col("peptide_id").replace_strict(same))
    )
    queries = queries.select(
        "query_id", "peptide_id", "set", "is_decoy", "sequence", "decoy_equals_target"
    )
    queries.write_csv(out / "queries.tsv", separator="\t")
    for name, part in [
        ("targets", queries.filter(~pl.col("is_decoy"))),
        ("decoys", queries.filter(pl.col("is_decoy"))),
    ]:
        with open(out / f"queries_{name}.fasta", "w") as fh:
            for qid, seq in part.select("query_id", "sequence").iter_rows():
                fh.write(f">{qid}\n{seq}\n")
    print(queries.group_by("set", "is_decoy").len().sort("set", "is_decoy"))
    print(f"{int(queries['decoy_equals_target'].sum())} decoys equal their peptide")


if __name__ == "__main__":
    main()
