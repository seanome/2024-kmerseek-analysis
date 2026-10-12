#!/usr/bin/env python3
"""Rebuild one search's queries.tsv from its region file, with the two awk errors fixed.

    repair_queries_tsv.py --queries in/X.queries.tsv --regions X.regions.csv.zst \
        --max-region-evalue 10000 --out X.queries.tsv

kmerseekSearch builds <slug>.queries.tsv (query_name, min_region_evalue) from the region
CSV with an awk that splits every line on a bare comma and compares E-values as `v+0`.
Both go wrong:

1. Names with a comma. Ten Botryllus FASTA ids hold one, e.g.
   FUN001269_BETA1,3GLUCANBINDING. kmerseek writes such a name in double quotes, so for
   its rows the awk stored '"FUN001269_BETA1', which is in no proteome, and, because
   query_name is the first column and every later field moves one place to the right,
   read region_ka_bits (22.4) where region_evalue (11_438) is. The search-time
   --max_region_evalue cut compared the same wrong column, so it kept such a row when
   region_ka_bits <= the cut. At the cut of 10_000 that is every row (a region scores far
   below 10_000 bits), so the region file still holds every row the right cut keeps.

2. "inf". kmerseek writes region_evalue = inf for some regions of an extend search. The
   image's awk reads the string "inf" as 0, so once a query has one inf row, that row wins
   the minimum, and every finite E-value of the query is lost: the queries.tsv says inf.
   In chunk_0000.dayhoff6.k11.s10.lcfalse.extend-c0.41 of the Botryllus midi run, 192 of
   1_570 queries were stored as inf, and 71 of them have a region with E <= 10.

This script reads the region file again and writes, for every query with any row, the
smallest region_evalue, comparing inf as infinity. Unquoted lines are split on commas
(only a query name can hold one: targets are bare accessions); quoted lines go through the
csv module. Finite E-values above --max-region-evalue are dropped, which only changes the
quoted rows (the search cut every other row right); empty and inf values are kept, as the
search kept them, so the set of queries with any region stays the same.

The fix lives here, after the search, because editing kmerseekSearch's script text would
rerun every search.
"""

import argparse
import csv
import math
import subprocess
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--regions", type=Path, required=True)
    ap.add_argument("--max-region-evalue", default="null",
                    help="the search's --max_region_evalue, or null when it ran with none")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    cut = None if args.max_region_evalue in ("null", "") else float(args.max_region_evalue)

    lines = args.queries.read_text().splitlines()
    header, body = lines[0], [ln for ln in lines[1:] if ln]
    if header.split("\t")[:2] != ["query_name", "min_region_evalue"]:
        sys.exit(f"{args.queries}: unexpected header {header!r}")
    old_names = {ln.split("\t")[0] for ln in body}

    best: dict[str, float] = {}
    explained: set[str] = set()   # the names the awk stored, for each name read here
    n_rows = n_quoted = n_cut = 0
    # -f: without it zstd skips a symbolic link with a warning and exits 0, and every file
    # Nextflow stages into a task is one.
    proc = subprocess.Popen(["zstd", "-dcf", str(args.regions)], stdout=subprocess.PIPE,
                            text=True, bufsize=1 << 20)
    first = proc.stdout.readline()
    if not first:
        sys.exit(f"{args.regions}: no header line")
    cols = next(csv.reader([first]))
    nc = len(cols)
    q = cols.index("query_name")
    e = cols.index("region_evalue") if "region_evalue" in cols else None
    if q != 0:
        sys.exit(f"{args.regions}: query_name is column {q}, this script needs it first")
    for line in proc.stdout:
        n_rows += 1
        if line.startswith('"'):
            n_quoted += 1
            row = next(csv.reader([line]))
            name = row[0]
            explained.add('"' + name.split(",")[0])
        else:
            row = line.rstrip("\n").split(",")
            name = row[0]
            explained.add(name)
        if len(row) != nc:
            sys.exit(f"{args.regions}: a row has {len(row)} fields, the header {nc}: "
                     f"{line[:200]!r}")
        raw = row[e] if e is not None else ""
        ev = float(raw) if raw else math.inf
        if cut is not None and math.isfinite(ev) and ev > cut:
            n_cut += 1
            continue
        if name not in best or ev < best[name]:
            best[name] = ev
    proc.stdout.close()
    if proc.wait() != 0:
        sys.exit(f"zstd -dcf {args.regions} exited {proc.returncode}")

    # Every name the awk wrote has to come from a row read here, and the other way round.
    if old_names != explained:
        sys.exit(f"{args.queries}: names differ from {args.regions}: only in the queries.tsv "
                 f"{sorted(old_names - explained)[:5]}, only in the region file "
                 f"{sorted(explained - old_names)[:5]}")

    out = [f"{n}\t{'inf' if math.isinf(v) else repr(v)}" for n, v in best.items()]
    args.out.write_text(header + "\n" + "".join(ln + "\n" for ln in sorted(out)))
    n_fixed = sum(1 for ln in body
                  if ln.split("\t")[1] == "inf" and ln.split("\t")[0] in best
                  and math.isfinite(best[ln.split("\t")[0]]))
    print(f"{args.queries.name}: {len(best)} queries from {n_rows} rows; "
          f"{sum(1 for n in old_names if n.startswith(chr(34)))} quoted names -> "
          f"{sum(1 for n in best if ',' in n)} ids with a comma ({n_quoted} rows); "
          f"{n_fixed} queries stored as inf now have a finite E-value; "
          f"{n_cut} rows above the E-value cut of {cut}")


if __name__ == "__main__":
    main()
