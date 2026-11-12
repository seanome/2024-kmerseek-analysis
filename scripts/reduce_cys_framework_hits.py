#!/usr/bin/env python3
"""Reduce a cys-frameworks run's kmerseek hit files to one row per query and target.

`kmerseek search` writes one row per matching region, so a pilot's nine hit files are
6 GB (2.8 GB for cys2 k12 alone). The notebook needs only the best region of each query
and target pair, which is a few MB, so this runs on Sherlock next to the results and the
laptop gets the small table.

For each `<level>.<clade>.<alphabet>.k<ksize>.hits.csv` it keeps, per query and target:
the smallest region E-value, the largest region bit score (Karlin-Altschul), the largest
whole-protein containment, and how many regions matched. It also writes the reference's
accessions, so the notebook can score against the same search space the searches saw.

Usage, on Sherlock, inside the pipeline image (it has polars):

    reduce_cys_framework_hits.py --results pilot/results --work pilot/work --out pilot/reduced
"""

from __future__ import annotations

import argparse
import gzip
import re
from pathlib import Path

import polars as pl

HITS_NAME = re.compile(
    r"^(?P<level>[^.]+)\.(?P<clade>[^.]+)\.(?P<alphabet>[^.]+)\.k(?P<ksize>\d+)\.hits\.csv$"
)


def reduce_hits(path: Path) -> pl.DataFrame:
    """One row per query and target accession, from one kmerseek hit file."""
    meta = HITS_NAME.match(path.name).groupdict()
    return (
        pl.scan_csv(path, infer_schema_length=10_000)
        .select(
            pl.col("query_name").alias("query_id"),
            # Swiss-Prot header `sp|ACC|NAME ...`: the accession is the second field.
            pl.col("target_name").str.split("|").list.get(1).alias("target_accession"),
            pl.col("region_evalue").cast(pl.Float64),
            pl.col("region_ka_bits").cast(pl.Float64),
            pl.col("containment").cast(pl.Float64),
            pl.col("n_intersecting_hashes").cast(pl.Int64),
            pl.col("query_poisson_pvalue").cast(pl.Float64),
            pl.col("region_poisson_score").cast(pl.Float64),
            pl.col("region_length").cast(pl.Int64),
            pl.col("region_subseq"),
            pl.col("target_subseq"),
            pl.col("moltype_seq"),
        )
        .group_by("query_id", "target_accession")
        .agg(
            pl.col("region_evalue").min().alias("best_region_evalue"),
            pl.col("region_ka_bits").max().alias("best_region_bits"),
            pl.col("containment").max().alias("containment"),
            pl.col("n_intersecting_hashes").max().alias("n_shared_kmers"),
            pl.col("query_poisson_pvalue").min().alias("query_poisson_pvalue"),
            pl.col("region_poisson_score").max().alias("best_region_poisson_score"),
            pl.col("region_length").max().alias("longest_region"),
            # The residues of the best region by Poisson score, so the notebook can show
            # what matched. Ties go to the longer region, then the earlier one in the file.
            *[
                pl.col(c)
                .sort_by(["region_poisson_score", "region_length"], descending=True)
                .first()
                .alias(f"best_{c}")
                for c in ("region_subseq", "target_subseq", "moltype_seq")
            ],
            pl.len().alias("n_regions"),
        )
        .with_columns(
            pl.lit(meta["level"]).alias("level"),
            pl.lit(meta["clade"]).alias("clade"),
            pl.lit(meta["alphabet"]).alias("alphabet"),
            pl.lit(int(meta["ksize"])).alias("ksize"),
        )
        .collect(engine="streaming")
        # A total sort key, so the file is the same on every run.
        .sort("query_id", "target_accession")
    )


def reference_accessions(fasta_gz: Path) -> list[str]:
    """Accessions in a held-out reference, from its `>sp|ACC|NAME` headers."""
    out = []
    with gzip.open(fasta_gz, "rt") as handle:
        for line in handle:
            if line.startswith(">"):
                out.append(line[1:].split("|")[1])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    for path in sorted((args.results / "kmerseek").glob("*.hits.csv")):
        table = reduce_hits(path)
        dest = args.out / path.name.replace(".hits.csv", ".pairs.parquet")
        table.write_parquet(dest)
        print(f"{path.name}: {table.height} query-target pairs -> {dest.name}")

    # The held-out references and their truth lists sit in heldOutReference's work
    # folders, next to the counts file that names the clade.
    for counts in sorted(args.work.glob("*/*/counts.*.tsv")):
        level, clade = counts.name.split(".")[1:3]
        accessions = reference_accessions(counts.parent / "reference.fasta.gz")
        (args.out / f"{level}.{clade}.reference_accessions.txt").write_text(
            "\n".join(accessions) + "\n"
        )
        truth = (counts.parent / "truth.txt").read_text()
        (args.out / f"{level}.{clade}.truth.txt").write_text(truth)
        print(f"{level}={clade}: {len(accessions)} reference entries")


if __name__ == "__main__":
    main()
