#!/usr/bin/env python3
"""The UniProt "CC -!- SIMILARITY" text (protein family and subfamily) of given Swiss-Prot
entries. Usage: similarity_lines.py <sprot.dat.gz> <accessions.txt> <out.csv>"""

import gzip
import sys

import polars as pl

dat, accs, out = sys.argv[1:4]
want = {l.strip() for l in open(accs) if l.strip()}
rows, acc, sim, in_sim = [], None, "", False
with gzip.open(dat, "rt") as f:
    for line in f:
        if line.startswith("AC   ") and acc is None:
            acc = line[5:].split(";")[0].strip()
        elif line.startswith("CC   -!- SIMILARITY:"):
            sim, in_sim = line[len("CC   -!- SIMILARITY:") :].strip(), True
        elif in_sim and line.startswith("CC       "):
            sim += " " + line[9:].strip()
        elif line.startswith("CC   ") or not line.startswith("CC"):
            in_sim = False
        if line.startswith("//"):
            if acc in want:
                rows.append({"accession": acc, "similarity": sim})
            acc, sim, in_sim = None, "", False
pl.DataFrame(rows).write_csv(out)
print(len(rows), "of", len(want))
