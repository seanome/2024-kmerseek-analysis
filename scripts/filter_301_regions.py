#!/usr/bin/env python3
"""Notebook 301: keep only the kmerseek regions the pass rule can use.

Standard library only. Reads a `kmerseek search` CSV (a file or stdin, so the search can be
piped straight in) and writes a gzipped TSV of the regions that
  * have region_evalue <= --evalue-max (10 by default), and
  * share at least one residue with a scored window of their query: a feature of a real
    query or chimera (truth.tsv), or the replaced residues of a decoy (decoys.tsv).
Every other row cannot change recall, boundary error or the decoy hit rate at any
threshold up to --evalue-max, so dropping it changes no number the rule reads.

Output columns: query_id, target_name, start, end (the region on the query, 1-based,
inclusive: kmerseek's 0-based end-exclusive region_start + 1 .. region_end),
region_length, region_evalue, region_ka_bits, target_start, target_end (0-based, as
written). A summary line on stderr counts rows read, kept and dropped.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import sys
from collections import defaultdict
from pathlib import Path

csv.field_size_limit(sys.maxsize)
KEEP = (
    "region_length",
    "region_evalue",
    "region_ka_bits",
    "target_start",
    "target_end",
)


def windows(truth: Path, decoys: Path) -> dict[str, list[tuple[int, int]]]:
    """Scored windows per query id, 1-based inclusive."""
    out: dict[str, list[tuple[int, int]]] = defaultdict(list)
    with open(truth) as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            out[r["query_id"]].append((int(r["start"]), int(r["end"])))
    with open(decoys) as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            if r["unchanged"] == "False":
                out[r["decoy_id"]].append((int(r["start"]), int(r["end"])))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--truth", type=Path, required=True)
    ap.add_argument("--decoys", type=Path, required=True)
    ap.add_argument("--search-csv", default="-")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--evalue-max", type=float, default=10.0)
    a = ap.parse_args()
    win = windows(a.truth, a.decoys)
    src = sys.stdin if a.search_csv == "-" else open(a.search_csv)
    n_read = n_evalue = n_kept = 0
    with gzip.open(a.out, "wt") as out:
        out.write("query_id\ttarget_name\tstart\tend\t" + "\t".join(KEEP) + "\n")
        for r in csv.DictReader(src):
            n_read += 1
            if not float(r["region_evalue"]) <= a.evalue_max:  # also drops nan
                continue
            n_evalue += 1
            q = r["query_name"].split()[0]
            s, e = int(r["region_start"]) + 1, int(r["region_end"])
            if not any(s <= we and e >= ws for ws, we in win.get(q, ())):
                continue
            n_kept += 1
            t = r["target_name"].split()[0]
            out.write(f"{q}\t{t}\t{s}\t{e}\t" + "\t".join(r[k] for k in KEEP) + "\n")
    print(
        f"rows read {n_read}, region_evalue <= {a.evalue_max:g} {n_evalue}, "
        f"kept (also on a scored window) {n_kept}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
