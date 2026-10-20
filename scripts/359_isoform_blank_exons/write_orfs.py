#!/usr/bin/env python3
"""Notebook 359: the ORFs InterProScan reads, one per coding isoform that passed the
SQANTI3 filter, named by isoform and taken from SQANTI3's own ORF_seq column, so the
InterProScan coordinates and build_exons.py's protein coordinates are on one sequence.

    write_orfs.py <classification.txt> <pass_isoforms.txt> <out.fasta>
"""

import csv
import sys

cls, passed, out = sys.argv[1:4]
keep = {line.strip() for line in open(passed) if line.strip()}
n = 0
with open(cls) as fh, open(out, "w") as fo:
    for r in csv.DictReader(fh, delimiter="\t"):
        seq = (r.get("ORF_seq") or "").rstrip("*")
        if r["isoform"] in keep and r["coding"] == "coding" and seq and seq not in ("NA",):
            fo.write(f">{r['isoform']}\n{seq}\n")
            n += 1
print(f"{n} ORFs written to {out}")
