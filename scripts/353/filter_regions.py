#!/usr/bin/env python3
"""Keep the kmerseek region rows notebook 353 counts, as the CSV streams past.

    kmerseek search ... | filter_regions.py <max_evalue> <out.tsv.gz> <summary.tsv>

A row is kept when region_evalue (the E-value kmerseek reports, Karlin-Altschul where the
pair has a fit, else the run E-value; see region_evalue_source) is at most <max_evalue>.
The largest cut-off notebook 353 reads is 10. kmerseek's own report filters have already
run; nothing about the search changes here. Adapted from notebook 344's
nextflow-runs/region-evalue-calibration/bin/filter_regions.py (PR 174). Standard library only.
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
    "region_length",
    "region_n_mismatches",
    "region_n_shared_kmers",
    "region_evalue",
    "region_evalue_source",
    "region_run_evalue",
    "region_ka_evalue",
]


def value(field):
    if field is None or field.strip() == "":
        return None
    x = float(field)
    return x if math.isfinite(x) else None


def main(max_evalue, out, summary):
    max_evalue = float(max_evalue)
    csv.field_size_limit(sys.maxsize)
    reader = csv.DictReader(sys.stdin)
    if reader.fieldnames is not None:
        missing = [c for c in KEEP if c not in reader.fieldnames]
        if missing:
            sys.exit(f"kmerseek output has no column(s) {missing}")
    rows_in = rows_kept = no_evalue = 0
    queries = set()
    with gzip.open(out, "wt", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(KEEP)
        for row in reader:
            rows_in += 1
            queries.add(row["query_name"])
            e = value(row["region_evalue"])
            if e is None:
                no_evalue += 1
            elif e <= max_evalue:
                rows_kept += 1
                w.writerow([row[c] for c in KEEP])
    with open(summary, "w") as fh:
        fh.write("rows_in\trows_kept\trows_no_evalue\tqueries_with_rows\n")
        fh.write(f"{rows_in}\t{rows_kept}\t{no_evalue}\t{len(queries)}\n")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    main(*sys.argv[1:])
