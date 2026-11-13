#!/usr/bin/env python3
"""Scan a held-out reference with the regular expression each query's cysteine spacing implies.

This is the control the experiment's verdict turns on. If this recovers the same homologs
as the best kmerseek setting for a framework, the spacing explains the performance and the
k-mers add nothing, so it runs on the same reference and is scored by the same rule as the
searches.

One row per (query, hit) pair, so the output reads like a search result.
"""

import argparse
import csv
import gzip
import re
import sys

# cys_framework_utils.py is staged into the task folder and found on PYTHONPATH, which
# the process sets. Python looks for imports next to the real file behind a staged
# script, which is bin/, not the task folder, so staging alone is not enough.
import cys_framework_utils as cfu  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-table", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--slack", type=int, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    with open(args.query_table) as handle:
        queries = list(csv.DictReader(handle, delimiter="\t"))

    # One compiled pattern per distinct spacing, not per query: queries of one framework
    # with the same spacer lengths share a pattern, and a reference scan is the slow part.
    patterns: dict[str, re.Pattern] = {}
    for row in queries:
        pattern = cfu.framework_regex(row["peptide"], slack=args.slack)
        row["regex"] = pattern
        if pattern not in patterns:
            patterns[pattern] = re.compile(pattern)
    print(
        f"{len(queries)} queries, {len(patterns)} distinct patterns at slack={args.slack}",
        file=sys.stderr,
    )

    hits: dict[str, list[str]] = {pattern: [] for pattern in patterns}
    n_reference = 0
    opener = gzip.open if args.reference.endswith(".gz") else open
    with opener(args.reference, "rt") as handle:
        header, chunks = None, []
        for line in handle:
            if line.startswith(">"):
                if header is not None:
                    n_reference += 1
                    sequence = "".join(chunks)
                    for pattern, compiled in patterns.items():
                        if compiled.search(sequence):
                            hits[pattern].append(header)
                header, chunks = line[1:].rstrip("\n"), []
            else:
                chunks.append(line.strip())
        if header is not None:
            n_reference += 1
            sequence = "".join(chunks)
            for pattern, compiled in patterns.items():
                if compiled.search(sequence):
                    hits[pattern].append(header)

    with open(args.out, "w", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(["query_id", "framework", "slack", "regex", "target_accession"])
        for row in queries:
            for header in hits[row["regex"]]:
                writer.writerow(
                    [
                        row["query_id"],
                        row["framework"],
                        args.slack,
                        row["regex"],
                        cfu.swissprot_header_fields(header)[0],
                    ]
                )

    total = sum(len(hits[row["regex"]]) for row in queries)
    print(
        f"scanned {n_reference} reference entries, {total} (query, hit) pairs",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
