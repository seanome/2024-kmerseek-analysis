#!/usr/bin/env python3
"""Check a tool's hits file before it is scored: the format every search module must write.

    check_tool_hits.py human_vs_zebrafish.diamond.tsv.gz \\
        [--query-fasta human.fasta --target-fasta zebrafish.fasta]

The format (CONTRIBUTING.md, "Output"): tab-separated, no header, gzip, one row per hit, and
these 8 columns in this order:

    query  target  qstart  qend  tstart  tend  score  evalue

query and target are the FASTA names as given to the tool (sp|P12345|NAME_SPECIES), so the
scorer can take the accession from them. Coordinates are 1-based and inclusive, start <= end,
on the query and target sequences as given. Higher score ranks first. evalue is the tool's
E-value, or for a tool that reports a probability, 1 - probability; it may be empty ("NA").
A motif tool may add a 9th column, n_matched_residues (a whole number), which the scorer reads.

Exits 1 and names the first bad rows when a row has the wrong number of columns, a
coordinate that is not a whole number, a start below 1, an end before its start, an end past
the sequence length (when the FASTA files are given), a score that is not a number, a
negative E-value, or a name not in the FASTA. An empty file also fails: no tool finds nothing
in 998 queries against a proteome, so an empty file means the command went wrong.
"""

import argparse
import gzip
import math
import sys
from pathlib import Path

N_COLS = (8, 9)
MAX_SHOWN = 5


def fasta_lengths(path: Path) -> dict[str, int]:
    opener = gzip.open if path.suffix == ".gz" else open
    lengths, name, n = {}, None, 0
    with opener(path, "rt") as f:
        for line in f:
            if line.startswith(">"):
                if name is not None:
                    lengths[name] = n
                name, n = line[1:].split()[0], 0
            else:
                n += len(line.strip())
    if name is not None:
        lengths[name] = n
    return lengths


def check(
    path: Path, qlen: dict[str, int] | None, tlen: dict[str, int] | None
) -> list[str]:
    problems: list[str] = []
    opener = gzip.open if path.suffix == ".gz" else open
    n_rows = 0

    def bad(i: int, msg: str) -> None:
        problems.append(f"row {i}: {msg}")

    with opener(path, "rt") as f:
        for i, line in enumerate(f, start=1):
            n_rows += 1
            if len(problems) >= MAX_SHOWN:
                break
            fields = line.rstrip("\n").split("\t")
            if len(fields) not in N_COLS:
                bad(
                    i,
                    f"{len(fields)} columns, expected 8 (or 9 for a motif tool): "
                    f"{line.strip()[:120]!r}",
                )
                continue
            if len(fields) == 9:
                if not fields[8].isdigit():
                    bad(i, f"n_matched_residues {fields[8]!r} is not a whole number")
                fields = fields[:8]
            q, t, *coords, score, evalue = fields
            try:
                qs, qe, ts, te = (int(x) for x in coords)
            except ValueError:
                bad(i, f"a coordinate is not a whole number: {coords}")
                continue
            if min(qs, ts) < 1:
                bad(i, f"coordinates are 1-based; got start {min(qs, ts)}")
            if qe < qs or te < ts:
                bad(
                    i,
                    f"an end comes before its start: query {qs}-{qe}, target {ts}-{te}",
                )
            try:
                s = float(score)
                if math.isnan(s):
                    raise ValueError
            except ValueError:
                bad(i, f"score {score!r} is not a number")
            if evalue not in ("", "NA"):
                try:
                    if float(evalue) < 0:
                        bad(i, f"E-value {evalue} is negative")
                except ValueError:
                    bad(i, f"E-value {evalue!r} is not a number")
            for name, end, lengths, side in (
                (q, qe, qlen, "query"),
                (t, te, tlen, "target"),
            ):
                if lengths is None:
                    continue
                if name not in lengths:
                    bad(i, f"{side} {name!r} is not in the {side} FASTA")
                elif end > lengths[name]:
                    bad(
                        i,
                        f"{side} end {end} is past the {lengths[name]}-residue sequence {name}",
                    )
    if n_rows == 0:
        problems.append("the file has no rows")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("hits", type=Path)
    ap.add_argument("--query-fasta", type=Path)
    ap.add_argument("--target-fasta", type=Path)
    a = ap.parse_args()
    qlen = fasta_lengths(a.query_fasta) if a.query_fasta else None
    tlen = fasta_lengths(a.target_fasta) if a.target_fasta else None
    problems = check(a.hits, qlen, tlen)
    for p in problems:
        print(f"{a.hits.name}: {p}", file=sys.stderr)
    if problems:
        print(
            f"{a.hits.name}: does not match the hits format in CONTRIBUTING.md",
            file=sys.stderr,
        )
        return 1
    print(f"{a.hits.name}: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
