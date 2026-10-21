#!/usr/bin/env python3
"""One row per Karlin-Altschul fit kmerseek stored in an index of the dark-set 0.4 store.

Usage: collect_ka_fits.py <kmerseek_index dir> <out.tsv>

Every index that was calibrated holds `ka_survival.C<penalty>.csv`; its rows all repeat the
fit's constants, so the first data row is enough. `slope` there is r_database, the factor a
search multiplies each pair's closed-form lambda by. `fitted` is false when kmerseek refused
the fit and stored nothing; those rows are kept so that refused settings are listed.
"""

import csv
import re
import sys
from pathlib import Path

NAME = re.compile(
    r"^(?P<database>[^.]+)\.(?P<alphabet>[a-z0-9_]+)\.k(?P<ksize>\d+)(?:\.s(?P<scaled>\d+))?"
    r"\.lc(?P<lowcomplexity>true|false)\.kmerseek\.rocksdb$"
)
KEEP = [
    "slope",
    "k",
    "lambda_analytic",
    "match_probability",
    "null",
    "reference",
    "mismatch_penalty",
    "xdrop",
    "n_queries",
    "query_residues",
    "database_kmers",
    "fitted",
]


def main() -> None:
    store, out = Path(sys.argv[1]), sys.argv[2]
    rows = []
    for survival in sorted(store.glob("*.rocksdb/ka_survival.C*.csv")):
        m = NAME.match(survival.parent.name)
        if m is None:
            raise ValueError(f"unexpected index name {survival.parent.name}")
        with open(survival) as fh:
            first = next(csv.DictReader(fh), None)
        if first is None:
            continue
        row = m.groupdict()
        row["scaled"] = row["scaled"] or "1"
        row.update({key: first[key] for key in KEEP})
        rows.append(row)
    with open(out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} fits from {store}")


if __name__ == "__main__":
    main()
