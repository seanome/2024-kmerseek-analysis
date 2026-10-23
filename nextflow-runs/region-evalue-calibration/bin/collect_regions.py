#!/usr/bin/env python3
"""Join the per-chunk files of one decoy kind into one, for notebook 344.

    collect_regions.py <kind> <out_dir> <chunk files...>

Each chunk is a pair <kind>.chunk_NNNN.regions.tsv.gz and <kind>.chunk_NNNN.summary.tsv
(from filter_regions.py). Writes <out_dir>/<kind>.regions.tsv.gz (one header, then every
chunk's rows, in chunk order) and <out_dir>/<kind>.summary.tsv (one row per chunk).
Stops if a chunk's row count differs from what its summary says it kept.
Standard library only.
"""

import csv
import gzip
import os
import sys


def main(kind, out_dir, *paths):
    regions = sorted(p for p in paths if p.endswith(".regions.tsv.gz"))
    summaries = {
        os.path.basename(p)[: -len(".summary.tsv")]: p
        for p in paths
        if p.endswith(".summary.tsv")
    }
    os.makedirs(out_dir, exist_ok=True)
    header, out_rows = None, []
    with gzip.open(
        os.path.join(out_dir, f"{kind}.regions.tsv.gz"), "wt", newline=""
    ) as out:
        w = csv.writer(out, delimiter="\t", lineterminator="\n")
        for p in regions:
            label = os.path.basename(p)[: -len(".regions.tsv.gz")]
            if not label.startswith(kind + "."):
                sys.exit(f"{p} is not a {kind} chunk")
            with gzip.open(p, "rt", newline="") as fh:
                r = csv.reader(fh, delimiter="\t")
                h = next(r)
                if header is None:
                    header = h
                    w.writerow(header)
                elif h != header:
                    sys.exit(f"{p} has different columns from the first chunk")
                n = 0
                for row in r:
                    w.writerow(row)
                    n += 1
            with open(summaries[label]) as fh:
                s = list(csv.DictReader(fh, delimiter="\t"))[0]
            if int(s["rows_kept"]) != n:
                sys.exit(f"{p}: {n} rows but its summary says {s['rows_kept']}")
            out_rows.append(s)
    if not out_rows:
        sys.exit(f"no {kind} chunks given")
    with open(os.path.join(out_dir, f"{kind}.summary.tsv"), "w") as fh:
        w = csv.DictWriter(
            fh, fieldnames=list(out_rows[0]), delimiter="\t", lineterminator="\n"
        )
        w.writeheader()
        w.writerows(out_rows)
    print(
        f"{kind}: {len(out_rows)} chunks, {sum(int(s['rows_kept']) for s in out_rows)} rows kept"
    )


if __name__ == "__main__":
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    main(*sys.argv[1:])
