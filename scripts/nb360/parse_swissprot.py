#!/usr/bin/env python3
"""Swiss-Prot flat file to a parquet table: accession, md5, genera, sequence.

`genera` holds every OC lineage term plus the first word of OS, so a genus test is a
list lookup (`Ciona`, `Danio`). md5 is of the upper-case sequence, the same as Pfam's
`md5` column, so a Pfam row applies to a Swiss-Prot entry only when both agree.

    parse_swissprot.py uniprot_sprot.dat.gz swissprot.parquet
"""

from __future__ import annotations

import gzip
import hashlib
import sys

import polars as pl


def main(dat: str, out: str) -> None:
    acc, lineage, os_word, seq, in_seq = None, [], None, [], False
    rows = []
    with gzip.open(dat, "rt") as fh:
        for line in fh:
            tag = line[:2]
            if tag == "AC" and acc is None:
                acc = line[5:].split(";")[0].strip()
            elif tag == "OS" and os_word is None:
                os_word = line[5:].split()[0]
            elif tag == "OC":
                lineage.extend(t.strip().rstrip(".") for t in line[5:].split(";") if t.strip())
            elif tag == "SQ":
                in_seq = True
            elif tag == "//":
                s = "".join(seq).upper()
                rows.append((acc, hashlib.md5(s.encode()).hexdigest(), sorted(set(lineage + [os_word])), s))
                acc, lineage, os_word, seq, in_seq = None, [], None, [], False
            elif in_seq:
                seq.append(line.replace(" ", "").strip())
    pl.DataFrame(rows, schema=["accession", "md5", "genera", "sequence"], orient="row").write_parquet(out)
    print(f"{len(rows)} Swiss-Prot entries -> {out}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
