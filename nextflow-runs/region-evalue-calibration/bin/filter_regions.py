#!/usr/bin/env python3
"""Keep the region rows notebook 344 counts, as kmerseek's CSV streams past.

    kmerseek search ... | filter_regions.py <max_evalue> <out.tsv.gz> <summary.tsv> <label>

The notebook counts regions whose E-value is at or below a cutoff. Its largest cutoff is
100 / (decoy queries in the smallest stratum), about 0.3, so a row is kept when any of
region_run_evalue, region_ka_evalue or region_evalue is <= <max_evalue> (1 by default in
main.nf). Every other row is counted in the summary and dropped. Nothing about the search
changes: kmerseek has already applied its own report filters and chosen its regions.

An empty field is a value kmerseek could not compute (kmerseek writes empty, never inf, for
those). "inf" and "nan" are read as not passing, in case an older build writes them.

<summary.tsv> has one row: label, rows_in, rows_kept, rows_ka_empty (rows with no
region_ka_evalue), queries_with_rows (distinct query_name among all rows).
Standard library only.
"""

import csv
import gzip
import math
import sys

KEEP = [
    "query_name",
    "target_name",
    "region_start",
    "region_end",
    "target_start",
    "target_end",
    "region_subseq",
    "target_subseq",
    "moltype_seq",
    "region_length",
    "region_n_mismatches",
    "region_n_shared_kmers",
    "region_n_chained",
    "region_run_length",
    "region_pr_same",
    "region_run_evalue",
    "region_ka_bits",
    "region_ka_evalue",
    "region_evalue",
    "region_evalue_source",
    "db_n_targets",
    "region_search_space",
]
EVALUES = ("region_run_evalue", "region_ka_evalue", "region_evalue")


def value(field):
    """The E-value as a float, or None when kmerseek left it empty or it is not finite."""
    if field is None or field.strip() == "":
        return None
    x = float(field)
    return x if math.isfinite(x) else None


def main(max_evalue, out, summary, label):
    max_evalue = float(max_evalue)
    csv.field_size_limit(sys.maxsize)
    reader = csv.DictReader(sys.stdin)
    # No header at all means kmerseek reported no regions for this chunk: write the
    # header-only outputs with rows_in = 0 (SEARCH has already checked kmerseek's exit code).
    missing = (
        [c for c in KEEP if c not in reader.fieldnames]
        if reader.fieldnames is not None
        else []
    )
    if missing:
        sys.exit(
            f"kmerseek output has no column(s) {missing}; is this the PR 133 build?"
        )
    rows_in = rows_kept = ka_empty = 0
    queries = set()
    with gzip.open(out, "wt", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(KEEP)
        for row in reader:
            rows_in += 1
            queries.add(row["query_name"])
            evalues = [value(row[c]) for c in EVALUES]
            if evalues[1] is None:
                ka_empty += 1
            if any(e is not None and e <= max_evalue for e in evalues):
                rows_kept += 1
                w.writerow([row[c] for c in KEEP])
    with open(summary, "w") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(
            ["label", "rows_in", "rows_kept", "rows_ka_empty", "queries_with_rows"]
        )
        w.writerow([label, rows_in, rows_kept, ka_empty, len(queries)])


if __name__ == "__main__":
    if len(sys.argv) != 5:
        sys.exit(__doc__)
    main(*sys.argv[1:])
