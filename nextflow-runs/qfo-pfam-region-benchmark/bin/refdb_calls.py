#!/usr/bin/env python3
"""Pfam calls from the reference-database tools: InterProScan and eggNOG-mapper.

These tools annotate the query against their own databases, which contain human, so they
are graded on their own models (as hmmscan is), not on transfer from a clade-excluded
reference. Output is the six-column form evaluate_domain_calls.py --direct-annotation
reads: query, pfam_id, qstart, qend, score, evalue (1-based, end included).

  pfam-names    Pfam-A.hmm -> NAME<TAB>ACC (bare PF id), to turn eggNOG's family names
                into accessions
  interproscan  InterProScan TSV -> its Pfam matches with their coordinates. The run has
                every member database on; the QC json counts matches per member database.
                Score is -log10 of the match E-value.
  eggnog        eggNOG-mapper .emapper.annotations -> one call per family in its PFAMs
                column. That column names families transferred from the seed ortholog
                and carries no coordinates, so each call spans the whole query (1..L) and
                the position metrics are "not applicable" for this tool. Score is the
                seed ortholog's bit score.
"""

import argparse
import gzip
import json
import math
import sys
from pathlib import Path

import polars as pl

OUT_SCHEMA = {
    "query": pl.String,
    "pfam_id": pl.String,
    "qstart": pl.Int64,
    "qend": pl.Int64,
    "score": pl.Float64,
    "evalue": pl.Float64,
}
# InterProScan 5 TSV columns (docs: "TSV" output format). 14 and 15 appear only with
# --goterms / --pathways.
IPS_COLS = [
    "protein",
    "md5",
    "length",
    "analysis",
    "signature",
    "signature_desc",
    "start",
    "stop",
    "score",
    "status",
    "date",
    "interpro",
    "interpro_desc",
    "go",
    "pathways",
]


def opener(path: Path):
    return gzip.open(path, "rt") if path.suffix == ".gz" else open(path)


def read_fasta_lengths(path: Path) -> dict[str, int]:
    lengths: dict[str, int] = {}
    name = None
    with opener(path) as fh:
        for line in fh:
            if line.startswith(">"):
                name = line[1:].split()[0]
                lengths[name] = 0
            elif name is not None:
                lengths[name] += len(line.strip())
    return lengths


def neg_log10(x: float | None) -> float | None:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    return 324.0 if x <= 0 else -math.log10(x)


def write_regions(df: pl.DataFrame, path: Path) -> None:
    df = df.select(list(OUT_SCHEMA)).sort("query", "qstart", "qend", "pfam_id")
    with gzip.open(path, "wt") as fh:
        fh.write(df.write_csv(separator="\t", include_header=False))


def cmd_pfam_names(a: argparse.Namespace) -> None:
    rows = []
    name = None
    with opener(a.pfam_hmm) as fh:
        for line in fh:
            if line.startswith("NAME "):
                name = line.split(None, 1)[1].strip()
            elif line.startswith("ACC ") and name is not None:
                rows.append((name, line.split(None, 1)[1].strip().split(".")[0]))
                name = None
    with open(a.out, "w") as fh:
        for n, acc in rows:
            fh.write(f"{n}\t{acc}\n")
    print(f"[pfam-names] {len(rows)} families", file=sys.stderr)


def read_interproscan(path: Path) -> pl.DataFrame:
    rows = []
    with opener(path) as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) < 11:
                continue
            rows.append(f[: len(IPS_COLS)] + [""] * (len(IPS_COLS) - len(f)))
    return pl.DataFrame(rows, schema=IPS_COLS, orient="row")


def cmd_interproscan(a: argparse.Namespace) -> None:
    ips = read_interproscan(a.tsv)
    lengths = read_fasta_lengths(a.queries)
    pfam = ips.filter(pl.col("analysis") == "Pfam")
    calls = pfam.select(
        pl.col("protein").alias("query"),
        pl.col("signature").str.split(".").list.get(0).alias("pfam_id"),
        pl.col("start").cast(pl.Int64).alias("qstart"),
        pl.col("stop").cast(pl.Int64).alias("qend"),
        pl.col("score").cast(pl.Float64, strict=False).alias("evalue"),
    ).with_columns(
        pl.col("evalue").map_elements(neg_log10, return_dtype=pl.Float64).alias("score")
    )
    write_regions(calls, a.out)
    per_member = (
        ips.group_by("analysis").len().sort("analysis").rows() if ips.height else []
    )
    qc = {
        "tool": "interproscan",
        "setting": a.setting,
        "n_queries": len(lengths),
        "n_queries_with_any_match": ips["protein"].n_unique(),
        "n_queries_with_pfam_match": pfam["protein"].n_unique(),
        "n_calls": calls.height,
        "n_pfam_matches_without_evalue": calls["evalue"].null_count(),
        "matches_per_member_database": {k: v for k, v in per_member},
        "queries_not_in_fasta": sorted(set(ips["protein"]) - set(lengths))[:10],
    }
    a.qc_out.write_text(json.dumps(qc, indent=1) + "\n")
    print(json.dumps(qc), file=sys.stderr)


def read_emapper(path: Path) -> pl.DataFrame:
    header = None
    rows = []
    with opener(path) as fh:
        for line in fh:
            if line.startswith("##"):
                continue
            if line.startswith("#"):
                header = line[1:].rstrip("\n").split("\t")
                continue
            if header is None:
                sys.exit(f"{path}: no '#query' header line before the first row")
            rows.append(line.rstrip("\n").split("\t"))
    if header is None:
        return pl.DataFrame(
            schema={
                "query": pl.String,
                "score": pl.String,
                "evalue": pl.String,
                "PFAMs": pl.String,
            }
        )
    return pl.DataFrame(rows, schema=header, orient="row")


def cmd_eggnog(a: argparse.Namespace) -> None:
    ann = read_emapper(a.annotations)
    lengths = read_fasta_lengths(a.queries)
    names = dict(
        line.rstrip("\n").split("\t") for line in open(a.pfam_names) if line.strip()
    )
    rows, unmatched = [], {}
    for r in ann.iter_rows(named=True):
        fams = [] if r["PFAMs"] in (None, "", "-") else r["PFAMs"].split(",")
        L = lengths.get(r["query"])
        if L is None:
            sys.exit(f"{r['query']} is in the annotations but not in {a.queries.name}")
        for fam in fams:
            acc = names.get(fam)
            if acc is None:
                unmatched[fam] = unmatched.get(fam, 0) + 1
                continue
            rows.append(
                {
                    "query": r["query"],
                    "pfam_id": acc,
                    "qstart": 1,
                    "qend": L,
                    "score": float(r["score"]),
                    "evalue": float(r["evalue"]),
                }
            )
    calls = pl.DataFrame(rows, schema=OUT_SCHEMA).unique(
        ["query", "pfam_id"], keep="first", maintain_order=True
    )
    write_regions(calls, a.out)
    qc = {
        "tool": "eggnog_mapper",
        "setting": a.setting,
        "n_queries": len(lengths),
        "n_queries_annotated": ann.height,
        "n_queries_with_pfam": calls["query"].n_unique(),
        "n_calls": calls.height,
        "call_interval": "whole query (1..length): PFAMs carries no coordinates",
        "n_family_names_not_in_pfam_hmm": sum(unmatched.values()),
        "family_names_not_in_pfam_hmm": dict(sorted(unmatched.items())[:50]),
    }
    a.qc_out.write_text(json.dumps(qc, indent=1) + "\n")
    print(json.dumps(qc), file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pfam-names")
    p.add_argument("--pfam-hmm", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=cmd_pfam_names)
    for name, func in (("interproscan", cmd_interproscan), ("eggnog", cmd_eggnog)):
        p = sub.add_parser(name)
        p.add_argument("--queries", type=Path, required=True)
        p.add_argument("--setting", required=True)
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--qc-out", type=Path, required=True)
        p.set_defaults(func=func)
        if name == "interproscan":
            p.add_argument("--tsv", type=Path, required=True)
        else:
            p.add_argument("--annotations", type=Path, required=True)
            p.add_argument("--pfam-names", type=Path, required=True)
    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
