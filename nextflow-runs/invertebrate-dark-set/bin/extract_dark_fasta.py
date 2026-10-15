#!/usr/bin/env python3
"""Write the dark proteins out of the query proteome as their own FASTA.

The Pfam arm scans the dark set, not the whole proteome: the placed proteins are placed
already and scanning them buys nothing the sequence arms have not said.

The accession is the whole first whitespace-delimited token, copied from
compute_dark_set.query_accessions -- deliberately NOT split on '|', because that would
break Botryllus's FUN000001_FUN000001 ids differently from UniProt's sp|P12345|NAME. It
has to stay the same rule: the dark parquet's `accession` column is the join key, and a
different rule here would silently drop every protein whose header it parsed differently.

Every dark accession must be found. A dark protein missing from the FASTA means the two
were built from different files, which is the failure this exits on rather than scanning a
short set and reporting a Pfam count against the wrong denominator.
"""
import argparse
import gzip
import sys
from pathlib import Path

import polars as pl


def accession_of(header: str) -> str:
    return header.split()[0]


def open_maybe_gzip(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "rt")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--query", type=Path, required=True, help="the staged query proteome")
    ap.add_argument("--dark", type=Path, required=True, help="<species>_dark_set.parquet")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    wanted = set(pl.read_parquet(args.dark)["accession"].to_list())
    if not wanted:
        # An empty dark set is a real answer, not an error: every protein was placed.
        args.out.write_text("")
        print("dark set is empty; wrote an empty FASTA", file=sys.stderr)
        return 0

    found: set[str] = set()
    written = 0
    with open_maybe_gzip(args.query) as fh, open(args.out, "w") as out:
        keep = False
        for line in fh:
            if line.startswith(">"):
                acc = accession_of(line[1:].strip())
                keep = acc in wanted
                if keep:
                    found.add(acc)
                    written += 1
            if keep:
                out.write(line)

    missing = wanted - found
    if missing:
        sample = ", ".join(sorted(missing)[:5])
        print(
            f"{len(missing)} of {len(wanted)} dark accessions are not in {args.query}.\n"
            f"First few: {sample}\n"
            "The dark parquet and this FASTA were built from different files, or their "
            "headers parse to different accessions. Scanning the short set would report a "
            "Pfam count against the wrong denominator, so this is a failure.",
            file=sys.stderr,
        )
        return 1

    print(f"wrote {written} dark proteins of {len(wanted)} to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
