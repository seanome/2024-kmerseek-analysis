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

With `--top-n N` it also writes, per hit file, the pairs a notebook needs to count toxins
in each query's top N: every pair in a score group (same best Poisson score and shared k-mer
count) that starts before place N, so a tie that crosses place N is kept whole
(`*.top<N>.parquet`), and per query the number of pairs and of pairs whose target is a
toxin (`*.per_query_counts.parquet`). A full phylum run's pairs tables are hundreds of MB;
these are what the laptop gets.

Usage, on Sherlock, inside the pipeline image (it has polars):

    reduce_cys_framework_hits.py --results pilot/results --work pilot/work --out pilot/reduced

`--hits-file NAME` reduces only that one hit file and skips the references, so each file can
be its own Slurm array task; `--references-only` writes only the references and truth lists.
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


def top_groups(pairs: pl.DataFrame, top_n: int) -> pl.DataFrame:
    """Pairs in the score groups that start before place `top_n` of their query's list."""
    key = ["best_region_poisson_score", "n_shared_kmers"]
    groups = (
        pairs.group_by("query_id", *key)
        .agg(n_tied=pl.len())
        .sort(["query_id", *key], descending=[False, True, True])
        .with_columns(
            n_before=(pl.col("n_tied").cum_sum() - pl.col("n_tied")).over("query_id")
        )
        .filter(pl.col("n_before") < top_n)
    )
    return pairs.join(groups, on=["query_id", *key], how="inner").sort(
        "query_id", "n_before", "target_accession"
    )


def per_query_counts(pairs: pl.DataFrame, truth: set[str]) -> pl.DataFrame:
    """Per query: pairs returned, and how many of them hit a toxin."""
    return (
        pairs.with_columns(toxin=pl.col("target_accession").is_in(list(truth)))
        .group_by("query_id")
        .agg(n_pairs=pl.len(), n_toxin_pairs=pl.col("toxin").sum())
        .sort("query_id")
    )


def find_counts(work: Path) -> dict[str, Path]:
    """heldOutReference work folders by `<level>.<clade>`, from their counts files."""
    return {
        ".".join(c.name.split(".")[1:3]): c.parent
        for c in sorted(work.glob("*/*/counts.*.tsv"))
    }


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
    parser.add_argument(
        "--hits-file", help="reduce only this file in <results>/kmerseek"
    )
    parser.add_argument("--references-only", action="store_true")
    parser.add_argument("--top-n", type=int, help="also write top-N groups and counts")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    folders = find_counts(args.work)

    paths = sorted((args.results / "kmerseek").glob("*.hits.csv"))
    if args.hits_file:
        paths = [args.results / "kmerseek" / args.hits_file]
    if args.references_only:
        paths = []
    for path in paths:
        table = reduce_hits(path)
        stem = path.name.replace(".hits.csv", "")
        table.write_parquet(args.out / f"{stem}.pairs.parquet")
        print(f"{path.name}: {table.height} query-target pairs -> {stem}.pairs.parquet")
        if args.top_n:
            meta = HITS_NAME.match(path.name).groupdict()
            truth = set(
                (folders[f"{meta['level']}.{meta['clade']}"] / "truth.txt")
                .read_text()
                .split()
            )
            top = top_groups(table, args.top_n)
            top.write_parquet(args.out / f"{stem}.top{args.top_n}.parquet")
            per_query_counts(table, truth).write_parquet(
                args.out / f"{stem}.per_query_counts.parquet"
            )
            print(f"  top {args.top_n} score groups: {top.height} pairs")
    if args.hits_file:
        return

    # The held-out references and their truth lists sit in heldOutReference's work
    # folders, next to the counts file that names the clade.
    for name, folder in folders.items():
        level, clade = name.split(".")
        accessions = reference_accessions(folder / "reference.fasta.gz")
        (args.out / f"{level}.{clade}.reference_accessions.txt").write_text(
            "\n".join(accessions) + "\n"
        )
        truth = (folder / "truth.txt").read_text()
        (args.out / f"{level}.{clade}.truth.txt").write_text(truth)
        print(f"{level}={clade}: {len(accessions)} reference entries")


if __name__ == "__main__":
    main()
