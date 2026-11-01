#!/usr/bin/env python3
"""Each query feature's identity to its closest regions-index entry of the same feature type.

Used to split recall by how far a feature is from anything in the regions index. It is
measured by one search that is not among the tools compared (MMseqs2 at -s 7.5 against the
regions index's targets, no decoys), so a feature's bin does not depend on whether a
compared tool found it.

Two modes:
  --write-fasta  writes each searched query feature's own residues, named by truth_id
  --read-hits    reads the MMseqs2 hits (query, target, pident, alnlen, evalue) and writes
                 feature_identity.tsv: truth_id, identity (best pident / 100 among hits
                 whose entry has the feature's type, E <= 10), identity_bin
"""

import argparse
from pathlib import Path

import polars as pl

BINS = [(0.3, "<30%"), (0.5, "30-50%"), (0.9, "50-90%"), (1.01, ">=90%")]
NONE = "no same-type entry found"


def read_fasta(path: Path) -> dict[str, str]:
    out, name, seq = {}, None, []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if name:
                    out[name] = "".join(seq)
                name, seq = line[1:].split()[0], []
            elif line:
                seq.append(line)
    if name:
        out[name] = "".join(seq)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--truth", type=Path, required=True)
    ap.add_argument("--searched", type=Path, required=True)
    ap.add_argument("--write-fasta", type=Path)
    ap.add_argument("--read-hits", type=Path)
    ap.add_argument("--out", type=Path, default=Path("feature_identity.tsv"))
    args = ap.parse_args()

    seqs = read_fasta(args.searched)
    truth = pl.read_parquet(args.truth).filter(pl.col("accession").is_in(list(seqs)))
    if args.write_fasta:
        with open(args.write_fasta, "w") as fh:
            for tid, acc, s, e in truth.select(
                "truth_id", "accession", "start", "end"
            ).iter_rows():
                fh.write(f">{tid}\n{seqs[acc][s - 1 : e]}\n")
        return

    hits = pl.read_csv(
        args.read_hits,
        separator="\t",
        has_header=False,
        new_columns=["query", "target", "pident", "alnlen", "evalue"],
        schema_overrides={"query": pl.UInt32, "target": pl.Utf8},
        quote_char=None,
    )
    best = (
        hits.with_columns(pl.col("target").str.split("|").list.get(1).alias("hit_type"))
        .join(
            truth.select(pl.col("truth_id").alias("query"), "feature_type"),
            on="query",
        )
        .filter(
            (pl.col("hit_type") == pl.col("feature_type")) & (pl.col("evalue") <= 10)
        )
        .group_by("query")
        .agg((pl.col("pident").max() / 100).alias("identity"))
        .rename({"query": "truth_id"})
    )
    expr = pl.lit(NONE)
    for hi, label in reversed(BINS):
        expr = pl.when(pl.col("identity") < hi).then(pl.lit(label)).otherwise(expr)
    out = (
        truth.select("truth_id")
        .join(best, on="truth_id", how="left")
        .with_columns(
            pl.when(pl.col("identity").is_null())
            .then(pl.lit(NONE))
            .otherwise(expr)
            .alias("identity_bin")
        )
    )
    out.write_csv(args.out, separator="\t")


if __name__ == "__main__":
    main()
