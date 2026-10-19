#!/usr/bin/env python3
"""Calls from the embedding searches, TM-Vec and PLMSearch, against a QfO target proteome.

Each human query's best hit (highest TM-Vec predicted TM-score, or highest PLMSearch
score) gives the Pfam domains of that target protein, from the target's domain map
(<species>_domain_map.parquet, the same map the main run's transfers read).

  short-headers  FASTA -> '>ACCESSION' headers (the UniProt accession, second
                 '|'-field), so ids survive tools that keep the whole header line
                 (PLMSearch's ESM loader) or split on spaces (TM-Vec)
  tmvec          TM-Vec tabular output, and optionally its DeepBLAST alignment output,
                 -> with alignments: the hits table annotator_best_hit.py place reads
                 (the domains are placed through the structural alignment); without:
                 whole-query family calls
  plmsearch      PLMSearch output (query, target, score per line) -> whole-query family
                 calls. PLMSearch reports no alignment, so its position scores are
                 "not applicable".

A whole-query call spans 1..L of the query: it names the family and claims no position.
"""

import argparse
import gzip
import json
import math
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import annotator_best_hit as abh  # noqa: E402

CALL_SCHEMA = {
    "query": pl.String,
    "pfam_id": pl.String,
    "qstart": pl.Int64,
    "qend": pl.Int64,
    "score": pl.Float64,
    "evalue": pl.Float64,
}


def accession(header_word: str) -> str:
    parts = header_word.split("|")
    return parts[1] if len(parts) >= 3 else header_word


def cmd_short_headers(a: argparse.Namespace) -> None:
    seqs = abh.read_fasta(a.fasta)
    seen = set()
    with open(a.out, "w") as fh:
        for name, seq in seqs.items():
            acc = accession(name)
            if acc in seen:
                sys.exit(f"{a.fasta.name}: accession {acc} appears twice")
            seen.add(acc)
            fh.write(f">{acc}\n{seq}\n")


def read_domain_map(path: Path) -> pl.DataFrame:
    return abh.read_domains(path)


def best_pairs(df: pl.DataFrame) -> pl.DataFrame:
    """(query, target, score) -> one row per query, highest score, ties to the smaller
    target id."""
    return (
        df.sort(["query", "score", "target"], descending=[False, True, False])
        .group_by("query", maintain_order=True)
        .first()
    )


def family_calls(
    best: pl.DataFrame, domains: pl.DataFrame, lengths: dict[str, int]
) -> pl.DataFrame:
    fams = domains.select("target", "pfam_id").unique()
    j = best.join(fams, on="target", how="inner")
    return j.select(
        "query",
        "pfam_id",
        pl.lit(1, dtype=pl.Int64).alias("qstart"),
        pl.col("query").replace_strict(lengths, return_dtype=pl.Int64).alias("qend"),
        pl.col("score").cast(pl.Float64),
        pl.lit(None, dtype=pl.Float64).alias("evalue"),
    ).sort("query", "pfam_id")


def write_calls(calls: pl.DataFrame, path: Path) -> None:
    with gzip.open(path, "wt") as fh:
        fh.write(
            calls.select(list(CALL_SCHEMA)).write_csv(
                separator="\t", include_header=False
            )
        )


def read_tmvec_tabular(path: Path) -> pl.DataFrame:
    t = pl.read_csv(path, separator="\t", infer_schema=False)
    t = t.rename({c: c.strip() for c in t.columns})
    return t.select(
        pl.col("query_id").alias("query"),
        pl.col("database_id").alias("target"),
        pl.col("tm-score").cast(pl.Float64).alias("score"),
    )


def read_deepblast(path: Path) -> list[tuple[str, str, str, str]]:
    """DeepBLAST alignment output: two lines per pair, 'name<spaces>gapped residues',
    query first. Returns (query, qaln, target, taln)."""
    lines = [ln.split() for ln in path.read_text().splitlines() if ln.strip()]
    if len(lines) % 2:
        sys.exit(f"{path.name}: odd number of alignment lines ({len(lines)})")
    return [(q[0], q[1], t[0], t[1]) for q, t in zip(lines[::2], lines[1::2])]


def alignment_hits(pairs, best: pl.DataFrame, qseqs: dict, tseqs: dict):
    """DeepBLAST pairs of each query's best target -> hits table rows. The start of an
    alignment is where its ungapped residues sit in the full sequence."""
    want = {(r["query"], r["target"]): r["score"] for r in best.iter_rows(named=True)}
    rows, unplaced = [], 0
    for q, qaln, t, taln in pairs:
        if (q, t) not in want:
            continue
        qs = qseqs[q].find(qaln.replace("-", ""))
        ts = tseqs[t].find(taln.replace("-", ""))
        if qs < 0 or ts < 0:
            unplaced += 1
            continue
        nq = len(qaln.replace("-", ""))
        nt = len(taln.replace("-", ""))
        rows.append(
            dict(
                query=q,
                target=t,
                qstart=qs + 1,
                qend=qs + nq,
                tstart=ts + 1,
                tend=ts + nt,
                bits=want[(q, t)],
                evalue=math.nan,
                pident=math.nan,
                qaln=qaln,
                taln=taln,
            )
        )
    return pl.DataFrame(rows, schema=abh.HIT_SCHEMA), unplaced


def cmd_tmvec(a: argparse.Namespace) -> None:
    best = best_pairs(read_tmvec_tabular(a.tabular))
    domains = read_domain_map(a.domain_map)
    qseqs = abh.read_fasta(a.queries)
    qc = {
        "tool": "tmvec",
        "setting": a.setting,
        "species": a.species,
        "n_queries": len(qseqs),
        "n_queries_with_hit": best.height,
    }
    if a.alignments is not None and a.alignments.stat().st_size > 0:
        hits, unplaced = alignment_hits(
            read_deepblast(a.alignments), best, qseqs, abh.read_fasta(a.targets)
        )
        calls, pq = abh.place_domains(hits, domains, a.min_overlap)
        calls = calls.select(list(CALL_SCHEMA))
        qc.update(pq)
        qc.update(
            {
                "call_interval": "placed through the DeepBLAST alignment",
                "n_best_hits_with_alignment": hits.height,
                "n_best_hits_without_alignment": best.height - hits.height - unplaced,
                "n_alignments_not_found_in_sequence": unplaced,
            }
        )
    else:
        calls = family_calls(best, domains, {k: len(v) for k, v in qseqs.items()})
        qc["call_interval"] = "whole query (1..length): no alignment"
    qc["n_queries_with_call"] = calls["query"].n_unique()
    qc["n_calls"] = calls.height
    write_calls(calls, a.out)
    a.qc_out.write_text(json.dumps(qc, indent=1) + "\n")
    print(json.dumps(qc), file=sys.stderr)


def cmd_plmsearch(a: argparse.Namespace) -> None:
    r = pl.read_csv(
        a.results,
        separator="\t",
        has_header=False,
        new_columns=["query", "target", "score"],
        schema_overrides={"query": pl.String, "target": pl.String, "score": pl.Float64},
    )
    best = best_pairs(r)
    domains = read_domain_map(a.domain_map)
    qseqs = abh.read_fasta(a.queries)
    calls = family_calls(best, domains, {k: len(v) for k, v in qseqs.items()})
    qc = {
        "tool": "plmsearch",
        "setting": a.setting,
        "species": a.species,
        "n_queries": len(qseqs),
        "n_queries_with_hit": best.height,
        "n_queries_with_call": calls["query"].n_unique(),
        "n_calls": calls.height,
        "call_interval": "whole query (1..length): PLMSearch reports no alignment",
    }
    write_calls(calls, a.out)
    a.qc_out.write_text(json.dumps(qc, indent=1) + "\n")
    print(json.dumps(qc), file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("short-headers")
    s.add_argument("--fasta", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    s.set_defaults(func=cmd_short_headers)
    for name, func in (("tmvec", cmd_tmvec), ("plmsearch", cmd_plmsearch)):
        p = sub.add_parser(name)
        p.add_argument("--queries", type=Path, required=True)
        p.add_argument("--domain-map", type=Path, required=True)
        p.add_argument("--setting", required=True)
        p.add_argument("--species", required=True)
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--qc-out", type=Path, required=True)
        p.set_defaults(func=func)
        if name == "tmvec":
            p.add_argument("--tabular", type=Path, required=True)
            p.add_argument("--alignments", type=Path, default=None)
            p.add_argument("--targets", type=Path, default=None)
            p.add_argument("--min-overlap", type=float, default=0.5)
        else:
            p.add_argument("--results", type=Path, required=True)
    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
