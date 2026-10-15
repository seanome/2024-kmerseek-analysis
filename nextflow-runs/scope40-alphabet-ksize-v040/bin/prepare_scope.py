#!/usr/bin/env python3
"""SCOP labels per domain, and the FASTA with each header cut to its domain id.

  prepare_scope.py <astral fasta> <domains.tsv> <ids.fa>

An ASTRAL header is `>d1dlwa_ a.1.1.1 (A:) Protein name {Species [TaxId: n]}`. The SCOP id
a.1.1.1 gives class a, fold a.1, superfamily a.1.1 and family a.1.1.1 (the same parse as
nextflow-runs/hp-alphabet-sweep). Fails on a header without a SCOP id or a repeated domain.
"""

import sys

src, out_tsv, out_fa = sys.argv[1:4]
seen = set()
with open(src) as fi, open(out_tsv, "w") as ft, open(out_fa, "w") as ff:
    ft.write("domain_id\tscop_id\tscop_class\tscop_fold\tscop_superfamily\tscop_family\n")
    for line in fi:
        if not line.startswith(">"):
            ff.write(line)
            continue
        parts = line[1:].split()
        if len(parts) < 2 or parts[1].count(".") != 3:
            sys.exit(f"no SCOP id in header: {line.strip()}")
        dom, sid = parts[0], parts[1]
        if dom in seen:
            sys.exit(f"domain {dom} appears twice")
        seen.add(dom)
        sp = sid.split(".")
        ft.write(f"{dom}\t{sid}\t{sp[0]}\t{'.'.join(sp[:2])}\t{'.'.join(sp[:3])}\t{sid}\n")
        ff.write(f">{dom}\n")
print(f"{len(seen)} domains", file=sys.stderr)
