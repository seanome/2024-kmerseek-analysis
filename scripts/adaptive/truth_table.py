#!/usr/bin/env python3
"""Step 2: one truth label per sampled read from its DIAMOND best hit against Swiss-Prot.

family      best hit carries the target Pfam family (Swiss-Prot DR Pfam lines, 2026_03)
non_family  best hit exists and does not carry it
unassigned  no hit at E <= 1e-5
Best hit = highest bitscore; ties broken by lower E-value, then accession, so the pick is
the same on every run.
"""

import argparse

import polars as pl

p = argparse.ArgumentParser()
p.add_argument("--diamond", required=True)
p.add_argument("--table", required=True)
p.add_argument(
    "--lengths-fastq",
    required=True,
    help="the sampled reads, for every read id and length",
)
p.add_argument("--pfam", required=True)
p.add_argument("--out", required=True)
a = p.parse_args()

ids, lens = [], []
with open(a.lengths_fastq) as f:
    for i, line in enumerate(f):
        if i % 4 == 0:
            ids.append(line[1:].split()[0])
        elif i % 4 == 1:
            lens.append(len(line.strip()))
reads = pl.DataFrame({"read_id": ids, "read_length": lens})

cols = "read_id sseqid pident aln_length mismatch gapopen qstart qend sstart send evalue bitscore qlen slen".split()
d = pl.read_csv(a.diamond, separator="\t", has_header=False, new_columns=cols)
best = (
    d.sort(
        ["read_id", "bitscore", "evalue", "sseqid"],
        descending=[False, True, False, False],
    )
    .unique("read_id", keep="first")
    .with_columns(best_hit=pl.col("sseqid").str.split("|").list.get(1))
)
t = pl.read_parquet(a.table).select(
    pl.col("accession").alias("best_hit"),
    pl.col("species").alias("best_hit_species"),
    pl.col("lineage").alias("best_hit_lineage"),
    pl.col("pfam").alias("best_hit_pfam"),
    pl.col("is_toxin_kw").alias("best_hit_is_toxin"),
)
out = (
    reads.join(
        best.select(
            "read_id",
            "best_hit",
            "pident",
            "aln_length",
            "qstart",
            "qend",
            "sstart",
            "send",
            "evalue",
            "bitscore",
        ),
        on="read_id",
        how="left",
    )
    .join(t, on="best_hit", how="left")
    .with_columns(
        label=pl.when(pl.col("best_hit").is_null())
        .then(pl.lit("unassigned"))
        .when(pl.col("best_hit_pfam").str.contains(a.pfam))
        .then(pl.lit("family"))
        .otherwise(pl.lit("non_family"))
    )
)
out.write_parquet(a.out)
print(out.group_by("label").len().sort("label"))
