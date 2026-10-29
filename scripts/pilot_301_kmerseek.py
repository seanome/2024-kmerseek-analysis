#!/usr/bin/env python3
"""Notebook 301 pilot: pick a few queries, then check what kmerseek wrote for them.

Standard library only (runs on Sherlock with `module load python/3.12.1`).

  select  writes pilot_queries.fasta: one real query per (organism group, feature type)
          present in truth.tsv, two chimeras (one signal peptide, one degron), and every
          decoy of those queries' features. Seed 301.
  check   reads one `kmerseek search` CSV for those queries and reports, per query kind:
          rows, rows whose region_evalue is finite, the smallest region_evalue, and
          whether region_subseq equals query[region_start:region_end] (0-based,
          end-exclusive; 1-based inclusive is region_start + 1 .. region_end).
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

SEED = 301
csv.field_size_limit(sys.maxsize)


def read_fasta(path: Path) -> dict[str, str]:
    seqs: dict[str, list[str]] = {}
    name = None
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith(">"):
                name = line[1:].split()[0]
                seqs[name] = []
            elif name is not None:
                seqs[name].append(line)
    return {k: "".join(v) for k, v in seqs.items()}


def read_tsv(path: Path) -> list[dict]:
    with open(path) as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def select(a) -> None:
    rng = random.Random(SEED)
    truth = read_tsv(a.step2 / "truth.tsv")
    decoys = read_tsv(a.step3 / "decoys.tsv")
    queries = read_fasta(a.step2 / "queries.fasta")
    decoy_seqs = read_fasta(a.step3 / "decoys.fasta")
    by_combo = defaultdict(set)
    for r in truth:
        if r["query_kind"] == "real":
            by_combo[(r["organism_group"], r["feature_type"])].add(r["query_id"])
    chosen = [rng.choice(sorted(v)) for k, v in sorted(by_combo.items())]
    chim = read_tsv(a.step2 / "chimeras.tsv")
    for kind in ("signal peptide", "degron"):
        chosen.append(
            rng.choice(sorted(r["query_id"] for r in chim if r["insert_kind"] == kind))
        )
    chosen = sorted(set(chosen))
    pick_decoys = [
        d for d in decoys if d["query_id"] in chosen and d["unchanged"] == "False"
    ]
    with open(a.out, "w") as fh:
        for q in chosen:
            fh.write(f">{q}\n{queries[q]}\n")
        for d in pick_decoys:
            fh.write(f">{d['decoy_id']}\n{decoy_seqs[d['decoy_id']]}\n")
    print(f"queries {len(chosen)}: {' '.join(chosen)}")
    print(f"decoys {len(pick_decoys)}")


def check(a) -> None:
    seqs = read_fasta(a.queries)
    stats = defaultdict(
        lambda: {"rows": 0, "finite": 0, "min_e": math.inf, "coord_ok": 0}
    )
    hit_queries = defaultdict(set)
    with open(a.search_csv) as fh:
        for r in csv.DictReader(fh):
            q = r["query_name"].split()[0]
            kind = (
                "decoy"
                if q.startswith("decoy_")
                else ("chimera" if q.startswith("chimera_") else "real")
            )
            s = stats[kind]
            s["rows"] += 1
            e = float(r["region_evalue"])
            if math.isfinite(e):
                s["finite"] += 1
                s["min_e"] = min(s["min_e"], e)
            st, en = int(r["region_start"]), int(r["region_end"])
            s["coord_ok"] += seqs[q][st:en] == r["region_subseq"]
            hit_queries[kind].add(q)
    n_kind = defaultdict(int)
    for q in seqs:
        n_kind[
            (
                "decoy"
                if q.startswith("decoy_")
                else ("chimera" if q.startswith("chimera_") else "real")
            )
        ] += 1
    print(
        "kind\tqueries\tqueries_with_rows\trows\trows_finite_region_evalue\tmin_region_evalue\trows_region_subseq_matches_query"
    )
    for kind in ("real", "chimera", "decoy"):
        s = stats[kind]
        print(
            f"{kind}\t{n_kind[kind]}\t{len(hit_queries[kind])}\t{s['rows']}\t{s['finite']}\t{s['min_e']:.3g}\t{s['coord_ok']}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--step2", type=Path, required=True)
    s.add_argument("--step3", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    c = sub.add_parser("check")
    c.add_argument("--queries", type=Path, required=True)
    c.add_argument("--search-csv", type=Path, required=True)
    a = ap.parse_args()
    select(a) if a.cmd == "select" else check(a)


if __name__ == "__main__":
    main()
