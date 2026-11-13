#!/usr/bin/env python3
"""Pfam seed sequences of the answer-key families, ungapped, for the distance control.

    seed_sequences.py Pfam-A.seed.gz answer_key.tsv seeds.fasta

Records are named <family>|<seed sequence name>, so a DIAMOND hit says which family it is
from. Pfam-A.seed is Stockholm format and Latin-1 encoded.
"""

from __future__ import annotations

import gzip
import sys

import polars as pl


def main(seed: str, key: str, out: str) -> None:
    want = set(pl.read_csv(key, separator="\t")["pfam_acc"].to_list())
    found: set[str] = set()
    fam, seqs = None, {}
    with gzip.open(seed, "rt", encoding="latin-1") as fh, open(out, "w") as o:
        for line in fh:
            if line.startswith("#=GF AC"):
                fam = line.split()[2].split(".")[0]
            elif line.startswith("//"):
                if fam in want:
                    found.add(fam)
                    for name, s in seqs.items():
                        o.write(f">{fam}|{name}\n{s}\n")
                fam, seqs = None, {}
            elif line.strip() and not line.startswith("#"):
                name, aln = line.split()
                s = "".join(ch for ch in aln if ch.isalpha()).upper()
                seqs[name] = seqs.get(name, "") + s
    print(f"{len(found)} of {len(want)} key families have seed sequences", file=sys.stderr)


if __name__ == "__main__":
    main(*sys.argv[1:4])
