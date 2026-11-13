#!/usr/bin/env python3
"""Which queries were already in a Pfam 33.1 seed alignment? (notebook 280, Stop A)

    seed_self_check.py --run-dir <RUN_DIR> --out <dir>

Reads the seed members make_search_inputs.py reference-pfam-seed dropped (a member whose
accession is a query, or whose sequence lies inside a query), finds every query that holds
each one and where, and joins the kept t1 instances of the pilot and full sets. An instance
is *seeded at t0* when a t0 seed member of the same family is that query's own sequence
(same accession, or the member's residues found inside the query) and the member's
interval on the query overlaps the t1 instance.

Writes dropped_members.csv (one row per dropped member and query holding it, with the
position on the query) and seeded_instances.csv (the instances seeded at t0).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_search_inputs import read_fasta, read_pfam_seed  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    r = a.run_dir
    a.out.mkdir(parents=True, exist_ok=True)

    mem = pl.read_csv(r / "inputs/reference-pfamseed33_1/seed_members.tsv.gz", separator="\t")
    dropped = mem.filter(pl.col("dropped").is_not_null() & (pl.col("dropped") != ""))
    want = set(dropped["member"])
    seqs = {
        f"{fam}__{acc}_{s}_{e}": seq
        for fam, acc, s, e, seq in read_pfam_seed(r / "data/Pfam33.1/Pfam-A.seed.gz")
        if f"{fam}__{acc}_{s}_{e}" in want
    }
    queries = {}
    for d in ("smoke", "pilot", "full"):
        queries.update(read_fasta(r / f"inputs/{d}/queries.fasta"))

    rows = []
    for m in dropped.iter_rows(named=True):
        seq = seqs[m["member"]]
        for q, qseq in queries.items():
            i = qseq.find(seq)
            while i >= 0:
                rows.append((m["member"], m["pfam_acc"], m["accession"], m["dropped"], len(seq),
                             q, i + 1, i + len(seq)))
                i = qseq.find(seq, i + 1)
        if m["dropped"] == "accession is a query" and seq not in queries[m["accession"]]:
            # Same accession, but the seed residues are not in the query's Pfam 33.1
            # sequence: placed by the seed's own start and end.
            rows.append((m["member"], m["pfam_acc"], m["accession"], m["dropped"], len(seq),
                         m["accession"], m["start"], m["end"]))
    hosts = pl.DataFrame(rows, orient="row", schema=[
        "member", "pfam_acc", "member_accession", "dropped", "member_length",
        "query", "on_query_start", "on_query_end"]).unique().sort("member", "query", "on_query_start")
    hosts = hosts.with_columns(
        is_decoy=pl.col("query").str.starts_with("DECOY_"),
        same_protein=pl.col("query") == pl.col("member_accession"),
    )
    hosts.write_csv(a.out / "dropped_members.csv")

    inst = pl.concat([
        pl.read_csv(r / f"inputs/{d}/instances.tsv", separator="\t")
        .select("query", "role", "pfam_acc", "pfam_type", "t1_start", "t1_end", "length")
        .with_columns(query_set=pl.lit(d))
        for d in ("pilot", "full")
    ])
    seeded = (
        inst.join(hosts.filter(~pl.col("is_decoy")), on=["query", "pfam_acc"], how="inner")
        .filter((pl.col("on_query_start") <= pl.col("t1_end")) & (pl.col("t1_start") <= pl.col("on_query_end")))
        .sort("query_set", "query", "t1_start")
    )
    seeded.write_csv(a.out / "seeded_instances.csv")

    with pl.Config(tbl_rows=100, tbl_width_chars=200):
        print(hosts.group_by("dropped", "is_decoy").agg(
            members=pl.col("member").n_unique(), queries=pl.col("query").n_unique()).sort("dropped", "is_decoy"))
        key = ["query", "pfam_acc", "t1_start", "t1_end"]
        for qs in ("pilot", "full"):
            n_all = inst.filter(pl.col("query_set") == qs).select(key).unique().height
            s = seeded.filter(pl.col("query_set") == qs)
            print(f"{qs}: {s.select(key).unique().height} of {n_all} instances seeded at t0, "
                  f"on {s['query'].n_unique()} queries; by how the member was found: "
                  f"{s.group_by('dropped').agg(pl.col('query').n_unique()).rows()}")


if __name__ == "__main__":
    main()
