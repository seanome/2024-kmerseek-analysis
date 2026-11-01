#!/usr/bin/env python3
"""List every ordered pair of different proteins in the same family, as FASTA header tokens.

family_pairs.py labels.tsv family.fasta family_pairs.tsv
"""

import sys

import polars as pl

labels, family_fasta, out = sys.argv[1:4]
family_of = dict(
    pl.read_csv(labels, separator="\t", infer_schema_length=0)
    .select("accession", "family")
    .unique()
    .iter_rows()
)
tokens = [line[1:].split()[0] for line in open(family_fasta) if line.startswith(">")]
acc = lambda tok: tok.split("|")[1]
missing = [acc(t) for t in tokens if acc(t) not in family_of]
assert not missing, f"no family in {labels} for {missing}"
with open(out, "w") as fh:
    for q in tokens:
        for t in tokens:
            if q != t and family_of[acc(q)] == family_of[acc(t)]:
                fh.write(f"{q}\t{t}\n")
