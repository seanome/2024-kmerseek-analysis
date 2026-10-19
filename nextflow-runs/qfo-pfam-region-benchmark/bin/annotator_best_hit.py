#!/usr/bin/env python3
"""Best-hit domain transfer: what most genome annotation pipelines do first.

Each human query is searched against Swiss-Prot with its own clade (Mammalia) removed.
The query's single best hit is kept. Every Pfam domain on that hit's protein is placed
on the query through the alignment, and each placed domain is a call.

Three subcommands, one per pipeline step:

  targets  the distinct best-hit proteins of every search, as a FASTA, so hmmscan can
           annotate only those (a few thousand proteins, not 570,000)
  place    best hits + the hit proteins' Pfam domains -> query-side domain calls, in the
           six-column form evaluate_domain_calls.py --direct-annotation reads
  (both share best_hits(), so the hit that gets annotated is the hit that gets placed)

Hits table columns (tab-separated, no header), the same for DIAMOND and MMseqs2:
    query target qstart qend tstart tend bits evalue pident qaln taln
qaln and taln are the aligned query and target residues with '-' for gaps (DIAMOND
qseq_gapped / sseq_gapped, MMseqs2 qaln / taln). All coordinates are 1-based with the end
included.

Best hit: highest bit score; ties go to the smaller E-value, then the smaller target
accession, so the choice never depends on file order.

Placement (the rule of PR 121, nb 280, made exact). For a target domain [ds, de] and a
hit aligning target tstart..tend, the covered part of the domain is
[max(tstart, ds), min(tend, de)]. The call is the span of query residues aligned to those
target residues. PR 121 read that span off the straight line between the hit's ends,
because its MMseqs2 and phmmer tables carried no alignment strings; here the alignment
is read residue by residue, so a gap inside the hit cannot move a boundary. The
straight-line placement is computed too, and `place` reports how many calls it would
change (qc json: n_calls_linear_differs, max_linear_shift).

A domain transfers only when the hit covers at least --min-overlap of it on the target,
the same rule evaluate_domain_calls.py applies to every other aligner in this pipeline.
"""

import argparse
import gzip
import json
import sys
from pathlib import Path

import polars as pl

HIT_COLS = [
    "query",
    "target",
    "qstart",
    "qend",
    "tstart",
    "tend",
    "bits",
    "evalue",
    "pident",
    "qaln",
    "taln",
]
HIT_SCHEMA = {
    "query": pl.String,
    "target": pl.String,
    "qstart": pl.Int64,
    "qend": pl.Int64,
    "tstart": pl.Int64,
    "tend": pl.Int64,
    "bits": pl.Float64,
    "evalue": pl.Float64,
    "pident": pl.Float64,
    "qaln": pl.String,
    "taln": pl.String,
}
# Target domains: hmmscan --domtblout reduced to these six columns by the pipeline.
DOMAIN_COLS = ["target", "pfam_id", "dstart", "dend", "dom_bits", "dom_ievalue"]


def read_hits(path: Path) -> pl.DataFrame:
    """A hits table; an empty file (no query had a hit) is an empty frame, not an error."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as fh:
        if not fh.read(1):
            return pl.DataFrame(schema=HIT_SCHEMA)
    return pl.read_csv(
        path,
        separator="\t",
        has_header=False,
        new_columns=HIT_COLS,
        schema_overrides=HIT_SCHEMA,
        quote_char=None,
    )


def best_hits(hits: pl.DataFrame) -> pl.DataFrame:
    """One row per query: its best hit (bits desc, evalue asc, target asc)."""
    return (
        hits.sort(
            ["query", "bits", "evalue", "target"],
            descending=[False, True, False, False],
        )
        .group_by("query", maintain_order=True)
        .first()
    )


def read_fasta(path: Path) -> dict[str, str]:
    """FASTA -> {first word of header: sequence}."""
    opener = gzip.open if path.suffix == ".gz" else open
    seqs: dict[str, list[str]] = {}
    name = None
    with opener(path, "rt") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith(">"):
                name = line[1:].split()[0]
                seqs[name] = []
            elif name is not None:
                seqs[name].append(line.strip())
    return {k: "".join(v) for k, v in seqs.items()}


def target_to_query_map(
    qstart: int, tstart: int, qaln: str, taln: str
) -> dict[int, int]:
    """Target residue -> the query residue aligned to it (gap columns map nothing)."""
    if len(qaln) != len(taln):
        raise ValueError(
            f"aligned strings differ in length ({len(qaln)} vs {len(taln)})"
        )
    m: dict[int, int] = {}
    q, t = qstart, tstart
    for a, b in zip(qaln, taln):
        q_res = a != "-"
        t_res = b != "-"
        if q_res and t_res:
            m[t] = q
        if q_res:
            q += 1
        if t_res:
            t += 1
    return m


def linear_place(qs: int, qe: int, ts: int, te: int, t: int) -> int:
    """PR 121's straight-line placement (score_transfers.project_domain)."""
    slope = (qe - qs) / (te - ts) if te > ts else 1.0
    return min(max(qs + round((t - ts) * slope), qs), qe)


def place_domains(
    best: pl.DataFrame, domains: pl.DataFrame, min_overlap: float
) -> tuple[pl.DataFrame, dict]:
    """Best hits x the hit protein's domains -> query-side calls, plus counts for QC."""
    rows = []
    n_below_min = 0
    n_no_aligned_residue = 0
    n_linear_differs = 0
    max_linear_shift = 0
    # group_by keys are 1-tuples.
    by_target = {k[0]: g for k, g in domains.group_by("target", maintain_order=True)}
    for h in best.iter_rows(named=True):
        doms = by_target.get(h["target"])
        if doms is None:
            continue
        amap = None
        for d in doms.iter_rows(named=True):
            lo = max(h["tstart"], d["dstart"])
            hi = min(h["tend"], d["dend"])
            dlen = d["dend"] - d["dstart"] + 1
            if hi < lo or (hi - lo + 1) / dlen < min_overlap:
                n_below_min += 1
                continue
            if amap is None:
                amap = target_to_query_map(
                    h["qstart"], h["tstart"], h["qaln"], h["taln"]
                )
            qpos = [amap[t] for t in range(lo, hi + 1) if t in amap]
            if not qpos:
                n_no_aligned_residue += 1
                continue
            qs, qe = min(qpos), max(qpos)
            ls = linear_place(h["qstart"], h["qend"], h["tstart"], h["tend"], lo)
            le = linear_place(h["qstart"], h["qend"], h["tstart"], h["tend"], hi)
            shift = max(abs(ls - qs), abs(le - qe))
            if shift:
                n_linear_differs += 1
                max_linear_shift = max(max_linear_shift, shift)
            rows.append(
                {
                    "query": h["query"],
                    "pfam_id": d["pfam_id"],
                    "qstart": qs,
                    "qend": qe,
                    "score": h["bits"],
                    "evalue": h["evalue"],
                    "target": h["target"],
                    "pident": h["pident"],
                    "linear_qstart": ls,
                    "linear_qend": le,
                }
            )
    calls = pl.DataFrame(
        rows,
        schema={
            "query": pl.String,
            "pfam_id": pl.String,
            "qstart": pl.Int64,
            "qend": pl.Int64,
            "score": pl.Float64,
            "evalue": pl.Float64,
            "target": pl.String,
            "pident": pl.Float64,
            "linear_qstart": pl.Int64,
            "linear_qend": pl.Int64,
        },
    )
    qc = {
        "n_domains_below_min_overlap": n_below_min,
        "n_domains_with_no_aligned_query_residue": n_no_aligned_residue,
        "n_calls": calls.height,
        "n_calls_linear_differs": n_linear_differs,
        "max_linear_shift": max_linear_shift,
    }
    return calls, qc


def read_domains(path: Path) -> pl.DataFrame:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as fh:
        if not fh.read(1):
            return pl.DataFrame(
                schema={
                    "target": pl.String,
                    "pfam_id": pl.String,
                    "dstart": pl.Int64,
                    "dend": pl.Int64,
                }
            )
    d = pl.read_csv(
        path,
        separator="\t",
        has_header=False,
        new_columns=DOMAIN_COLS,
        schema_overrides={"target": pl.String, "pfam_id": pl.String},
    )
    # hmmscan names versioned accessions (PF00001.24); the truth keys on the bare id.
    return d.select(
        "target",
        pl.col("pfam_id").str.split(".").list.get(0),
        pl.col("dstart").cast(pl.Int64),
        pl.col("dend").cast(pl.Int64),
    ).sort("target", "dstart", "dend", "pfam_id")


def cmd_targets(a: argparse.Namespace) -> None:
    targets: set[str] = set()
    for p in a.hits:
        targets.update(best_hits(read_hits(p))["target"].to_list())
    seqs = read_fasta(a.reference)
    missing = sorted(t for t in targets if t not in seqs)
    if missing:
        sys.exit(
            f"{len(missing)} best-hit targets are not in {a.reference.name}, e.g. "
            f"{missing[:3]}: the search ran against a different database"
        )
    with open(a.out, "w") as fh:
        for t in sorted(targets):
            fh.write(f">{t}\n{seqs[t]}\n")
    print(f"[targets] {len(targets)} distinct best-hit proteins", file=sys.stderr)


def cmd_place(a: argparse.Namespace) -> None:
    hits = read_hits(a.hits)
    best = best_hits(hits)
    domains = read_domains(a.target_domains)
    calls, qc = place_domains(best, domains, a.min_overlap)
    n_queries = len(read_fasta(a.queries)) if a.queries else None
    hit_targets = set(best["target"].to_list())
    with_domain = hit_targets & set(domains["target"].to_list())
    qc = {
        "tool": a.tool,
        "setting": a.setting,
        "n_queries": n_queries,
        "n_queries_with_hit": best.height,
        "n_best_hits_with_pfam_domain": best.filter(
            pl.col("target").is_in(list(with_domain))
        ).height,
        "n_queries_with_call": calls["query"].n_unique(),
        "min_overlap": a.min_overlap,
        **qc,
    }
    # Six columns, no header: query, pfam_id, qstart, qend, score, evalue -- the
    # --direct-annotation form (same as hmmscanAnnotate's output).
    out = calls.select("query", "pfam_id", "qstart", "qend", "score", "evalue").sort(
        "query", "qstart", "qend", "pfam_id"
    )
    # Written as one string: polars' write_csv into a gzip text handle produced a stream
    # zlib could not read back ("invalid distance too far back"), found on the fixture.
    with gzip.open(a.out, "wt") as fh:
        fh.write(out.write_csv(separator="\t", include_header=False))
    if a.detail_out:
        calls.write_parquet(a.detail_out)
    a.qc_out.write_text(json.dumps(qc, indent=1) + "\n")
    print(json.dumps(qc), file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("targets", help="FASTA of every search's best-hit proteins")
    t.add_argument("--hits", type=Path, nargs="+", required=True)
    t.add_argument("--reference", type=Path, required=True)
    t.add_argument("--out", type=Path, required=True)
    t.set_defaults(func=cmd_targets)
    p = sub.add_parser("place", help="place the best hit's domains on the query")
    p.add_argument("--hits", type=Path, required=True)
    p.add_argument("--target-domains", type=Path, required=True)
    p.add_argument("--queries", type=Path, default=None)
    p.add_argument("--tool", required=True)
    p.add_argument("--setting", required=True)
    p.add_argument("--min-overlap", type=float, default=0.5)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--detail-out", type=Path, default=None)
    p.add_argument("--qc-out", type=Path, required=True)
    p.set_defaults(func=cmd_place)
    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
