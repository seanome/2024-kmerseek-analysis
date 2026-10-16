#!/usr/bin/env python3
"""Signal peptide, propeptide and mature chain ranges (Swiss-Prot FT lines) for the
proteins of a reference FASTA. Usage: reference_features.py <sprot.dat.gz> <ref.fasta> <out.csv>
"""

import gzip
import sys

import polars as pl

dat, ref, out = sys.argv[1:4]
want = {l[1:].strip() for l in open(ref) if l.startswith(">")}
rows, acc = [], None
with gzip.open(dat, "rt") as f:
    for line in f:
        if line.startswith("AC   ") and acc is None:
            acc = line[5:].split(";")[0].strip()
        elif line.startswith("FT   ") and acc in want:
            p = line[5:].split()
            if p and p[0] in ("SIGNAL", "PROPEP", "CHAIN", "PEPTIDE") and ".." in p[1]:
                a, b = p[1].split("..")
                if a.isdigit() and b.isdigit():
                    rows.append(
                        {
                            "accession": acc,
                            "feature": p[0],
                            "start": int(a),
                            "end": int(b),
                        }
                    )
        elif line.startswith("//"):
            acc = None
pl.DataFrame(rows).write_csv(out)
print(
    len(rows),
    "features for",
    len({r["accession"] for r in rows}),
    "of",
    len(want),
    "proteins",
)
