#!/usr/bin/env python3
"""Recount the query-strata bins from the raw job outputs, and check the long table.

Written separately from build_query_strata.py and sharing no code with it or with the
scorer: it parses the raw MMseqs2 hit tables (every hit, not the best-hit file), the
segmasker intervals, DeepTMHMM's 3-line file, BUSCO's full table and the query FASTA
itself, rebuilds every bin with plain Python, and compares:

  1. per query, its bin on every protein-level axis against query_strata.tsv and
     query_nearest_hits.tsv;
  2. per axis, the number of queries in each bin;
  3. in results.parquet, for every (experiment, tool, setting, query set, target, truth,
     dedup, metric) and every query-side axis, that n_instances summed over the axis's
     bins equals the whole-set row (family axes: Pfam truth sets only).

Writes results/leaderboard/query_strata/checks.md and exits non-zero on any mismatch.
"""

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import polars as pl

QS = "leaderboard-query-strata/out"
QUERY_AXES = [
    "nearest_identity",
    "membrane",
    "low_complexity",
    "protein_length",
    "region_length",
    "family_superkingdoms",
    "family_phyla",
    "busco",
]
FAMILY_AXES = {"family_superkingdoms", "family_phyla"}
PFAM_TRUTH = {"Pfam-A domain instances", "Pfam-N domain instances"}


def acc_of(name):
    f = name.split("|")
    return f[1] if len(f) >= 3 else name


def fasta_lengths(path):
    out, cur = {}, None
    for line in open(path):
        line = line.strip()
        if line.startswith(">"):
            cur = acc_of(line[1:].split()[0])
            out[cur] = 0
        elif cur:
            out[cur] += len(line)
    return out


def best_hits(m8_dir):
    """species -> accession -> identity percent of the best hit at E < 1e-3."""
    out = {}
    for f in sorted(Path(m8_dir).glob("*.m8")):
        best = {}
        for row in csv.reader(open(f), delimiter="\t"):
            q, t, fid, ev, bits = (
                row[0],
                row[1],
                float(row[2]),
                float(row[10]),
                float(row[11]),
            )
            if ev >= 1e-3:
                continue
            key = (ev, -bits, t)
            a = acc_of(q)
            if a not in best or key < best[a][0]:
                best[a] = (key, fid * 100 if fid <= 1 else fid)
        out[f.stem] = {a: v[1] for a, v in best.items()}
    return out


def ident_bin(x):
    if x is None:
        return "no hit"
    return "<30%" if x < 30 else "30-50%" if x < 50 else "50-90%" if x < 90 else ">=90%"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument(
        "--strata", type=Path, default=Path("results/leaderboard/query_strata")
    )
    ap.add_argument(
        "--results", type=Path, default=Path("results/leaderboard/results.parquet")
    )
    args = ap.parse_args()
    d = args.data / QS
    lines, bad = [], 0

    lengths = fasta_lengths(d / "deeptmhmm/queries.fasta")
    accs = sorted(lengths)

    masked = defaultdict(int)
    cur = None
    for line in open(d / "seg/queries.seg.intervals"):
        if line.startswith(">"):
            cur = acc_of(line[1:].split()[0])
        elif line.strip():
            a, b = line.split("-")
            masked[cur] += int(b) - int(a) + 1
    dtm = {}
    for line in open(d / "deeptmhmm/predicted_topologies.3line"):
        if line.startswith(">"):
            name, typ = line[1:].rsplit("|", 1)
            dtm[acc_of(name.strip())] = typ.strip()
    busco = {}
    for line in open(d / "busco/busco_human/run_primates_odb12/full_table.tsv"):
        if line.startswith("#"):
            continue
        f = line.rstrip("\n").split("\t")
        if len(f) > 2 and f[1] in ("Complete", "Duplicated", "Fragmented"):
            a = acc_of(f[2].split()[0])
            busco[a] = (
                "complete"
                if (f[1] != "Fragmented" or busco.get(a) == "complete")
                else "fragmented"
            )

    mine = {}
    for a in accs:
        n, fr = lengths[a], masked[a] / lengths[a]
        mine[a] = {
            "membrane": {
                "TM": "alpha-helical TM",
                "SP+TM": "alpha-helical TM",
                "BETA": "beta-barrel",
            }.get(dtm[a], "neither"),
            "low_complexity": "<5%" if fr < 0.05 else "5-20%" if fr <= 0.2 else ">20%",
            "protein_length": (
                "<100"
                if n < 100
                else "100-300" if n < 300 else "300-1000" if n <= 1000 else ">1000"
            ),
            "busco": busco.get(a, "not a BUSCO gene"),
        }
    built = {
        r["accession"]: r
        for r in pl.read_csv(
            args.strata / "query_strata.tsv", separator="\t"
        ).iter_rows(named=True)
    }
    lines.append("## Protein-level bins, recounted from the raw outputs\n")
    lines.append(
        "| axis | bin | queries (recount) | queries (query_strata.tsv) | per-query match |"
    )
    lines.append("|---|---|---:|---:|---|")
    for axis in ("membrane", "low_complexity", "protein_length", "busco"):
        c1 = Counter(mine[a][axis] for a in accs)
        c2 = Counter(built[a][axis] for a in accs)
        agree = all(mine[a][axis] == built[a][axis] for a in accs)
        bad += (c1 != c2) + (not agree)
        for b in sorted(set(c1) | set(c2)):
            lines.append(
                f"| {axis} | {b} | {c1[b]} | {c2[b]} | {'yes' if agree else 'NO'} |"
            )

    hits = best_hits(d / "identity")
    near = pl.read_csv(args.strata / "query_nearest_hits.tsv", separator="\t")
    near_bin = {
        (r["species"], r["accession"]): r["nearest_identity"]
        for r in near.iter_rows(named=True)
    }
    lines.append("\n## Nearest-protein identity, recounted from every MMseqs2 hit\n")
    lines.append(
        "| target | no hit | <30% | 30-50% | 50-90% | >=90% | per-query match |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---|")
    for sp in sorted(hits):
        c = Counter(ident_bin(hits[sp].get(a)) for a in accs)
        agree = all(ident_bin(hits[sp].get(a)) == near_bin[(sp, a)] for a in accs)
        bad += not agree
        lines.append(
            f"| {sp} | "
            + " | ".join(
                str(c[b]) for b in ["no hit", "<30%", "30-50%", "50-90%", ">=90%"]
            )
            + f" | {'yes' if agree else 'NO'} |"
        )

    lines.append("\n## Long table: bins add up to the whole set\n")
    if args.results.exists():
        r = pl.read_parquet(args.results)
        q = r.filter(pl.col("stratum_axis").is_in(QUERY_AXES))
        if q.height == 0:
            lines.append(
                "results.parquet has no query-side rows yet: the midi-plus run has "
                "not been rescored with the strata covariates."
            )
        else:
            key = [
                "experiment",
                "tool",
                "setting",
                "query_set",
                "target_db",
                "truth_source",
                "dedup_or_raw",
                "metric",
            ]
            whole = r.filter(pl.col("stratum_axis") == "all").select(
                *key, pl.col("n_instances").alias("whole")
            )
            sums = (
                q.group_by(*key, "stratum_axis")
                .agg(pl.col("n_instances").sum().alias("sum"))
                .filter(
                    ~pl.col("stratum_axis").is_in(list(FAMILY_AXES))
                    | pl.col("truth_source").is_in(list(PFAM_TRUTH))
                )
                .join(whole, on=key, how="left")
            )
            wrong = sums.filter(pl.col("sum") != pl.col("whole"))
            bad += wrong.height
            lines.append(
                f"{sums.height:,} (row group, axis) pairs checked; "
                f"{wrong.height:,} do not add up."
            )
    else:
        lines.append("results.parquet not found.")

    lines.append(f"\nMismatches: {bad}.")
    (args.strata / "checks.md").write_text(
        "# Query strata checks\n\n" + "\n".join(lines) + "\n"
    )
    print("\n".join(lines))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
