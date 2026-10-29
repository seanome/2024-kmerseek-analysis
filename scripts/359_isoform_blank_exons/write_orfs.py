#!/usr/bin/env python3
"""Notebook 359: the ORFs InterProScan reads, one per coding isoform that passed the
SQANTI3 filter, named by isoform and taken from SQANTI3's own ORF_seq column, so the
InterProScan coordinates and build_exons.py's protein coordinates are on one sequence.

    write_orfs.py <classification.txt> <pass_isoforms.txt> <out.fasta>
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "notebooks"))
import isoform_blank_exons_359 as ib  # noqa: E402

cls, passed, out = sys.argv[1:4]
keep = {line.strip() for line in open(passed) if line.strip()}
n = 0
with open(cls) as fh, open(out, "w") as fo:
    for r in csv.DictReader(fh, delimiter="\t"):
        raw = r.get("ORF_seq") or ""
        seq = None
        if r["isoform"] in keep and r["coding"] == "coding" and raw not in ("", "NA"):
            # the same filter as build_exons.py: length matches CDS, no internal stop
            seq = ib.usable_orf(raw, r["CDS_start"], r["CDS_end"])
        if seq:
            fo.write(f">{r['isoform']}\n{seq}\n")
            n += 1
print(f"{n} ORFs written to {out}")
