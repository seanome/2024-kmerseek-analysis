#!/usr/bin/env python3
"""Write one held-out reference, its truth list and the queries that belong to it.

Held out means Swiss-Prot with one clade removed, so no hit a query gets can come from
its own genus or its own phylum. An entry whose taxid has no rank at that level stays in,
since nothing places it inside the excluded clade.
"""

import argparse
import csv
import gzip
import sys

# cys_framework_utils.py is staged into the task folder and found on PYTHONPATH, which
# the process sets. Python looks for imports next to the real file behind a staged
# script, which is bin/, not the task folder, so staging alone is not enough.
import cys_framework_utils as cfu  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--reference-entries", required=True)
    parser.add_argument("--swissprot", required=True)
    parser.add_argument("--level", required=True, choices=["genus", "phylum"])
    parser.add_argument("--excluded", required=True)
    parser.add_argument("--out-fasta", required=True)
    parser.add_argument("--out-truth", required=True)
    parser.add_argument("--out-queries", required=True)
    parser.add_argument("--out-query-table", required=True)
    parser.add_argument("--out-counts", required=True)
    parser.add_argument(
        "--sample-non-toxins",
        type=int,
        default=0,
        help=(
            "Pilot only. Keep every Toxin-keyword entry but only this many others, chosen "
            "by a seeded hash of the header so the choice is the same on every run. 0 "
            "keeps all of Swiss-Prot, which is what a real run does. A sampled reference "
            "makes precision look better than it is, because most of the entries a search "
            "could wrongly hit are not there, so it proves the pipeline runs and nothing "
            "more."
        ),
    )
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument(
        "--queries-per-framework",
        type=int,
        default=0,
        help="Pilot only. Keep at most this many queries per framework; 0 keeps all.",
    )
    args = parser.parse_args()

    # reference_entries.csv carries the taxid and the Toxin flag per header, worked out
    # once by scripts/build_cys_framework_data.py, so this does not re-resolve taxonomy.
    excluded_headers = set()
    toxin_headers = set()
    header_level = {}
    with open(args.reference_entries) as handle:
        for row in csv.DictReader(handle):
            header_level[row["header"]] = row[args.level]
            if row["is_toxin"] == "true":
                toxin_headers.add(row["header"])
            if row[args.level] == args.excluded:
                excluded_headers.add(row["header"])

    sampled_out = set()
    if args.sample_non_toxins:
        import hashlib

        others = [h for h in header_level if h not in toxin_headers]
        others.sort(
            key=lambda h: hashlib.sha256(f"{args.sample_seed}:{h}".encode()).hexdigest()
        )
        sampled_out = set(others[args.sample_non_toxins :])
        print(
            f"pilot sampling: keeping {len(others) - len(sampled_out)} of {len(others)} "
            "non-toxin entries",
            file=sys.stderr,
        )

    kept = 0
    removed = 0
    toxins_left = 0
    with gzip.open(args.swissprot, "rt") as src, gzip.open(args.out_fasta, "wt") as out:
        writing = False
        for line in src:
            if line.startswith(">"):
                header = line[1:].rstrip("\n")
                writing = header not in excluded_headers and header not in sampled_out
                if writing:
                    kept += 1
                    if header in toxin_headers:
                        toxins_left += 1
                elif header in excluded_headers:
                    removed += 1
            if writing:
                out.write(line)

    with open(args.out_truth, "w") as out:
        for header in sorted(toxin_headers - excluded_headers):
            out.write(cfu.swissprot_header_fields(header)[0] + "\n")

    # The queries this reference is for: those whose own clade is the excluded one.
    rows = []
    with open(args.queries) as handle:
        for row in csv.DictReader(handle):
            if row[args.level] == args.excluded:
                rows.append(row)
    if args.queries_per_framework:
        by_framework: dict[str, list[dict]] = {}
        for row in rows:
            by_framework.setdefault(row["framework"], []).append(row)
        rows = [
            row
            for framework in sorted(by_framework)
            for row in sorted(by_framework[framework], key=lambda r: r["query_id"])[
                : args.queries_per_framework
            ]
        ]
        print(
            f"pilot: {len(rows)} queries, at most {args.queries_per_framework} per "
            f"framework over {len(by_framework)} frameworks",
            file=sys.stderr,
        )
    with open(args.out_queries, "w") as out:
        for row in rows:
            out.write(f">{row['query_id']}\n{row['peptide']}\n")
    with open(args.out_query_table, "w") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    with open(args.out_counts, "w") as out:
        writer = csv.writer(out, delimiter="\t")
        writer.writerow(
            [
                "level",
                "excluded",
                "n_reference",
                "n_removed",
                "n_toxins_left",
                "n_queries",
            ]
        )
        writer.writerow(
            [args.level, args.excluded, kept, removed, toxins_left, len(rows)]
        )

    print(
        f"{args.level}={args.excluded}: reference {kept} entries "
        f"({removed} removed, {toxins_left} toxins left), {len(rows)} queries",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
