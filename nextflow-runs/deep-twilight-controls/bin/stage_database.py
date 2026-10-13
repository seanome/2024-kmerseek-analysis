#!/usr/bin/env python3
"""Build the search database: the human proteome plus the 16 family proteins.

The human family members (8 of the 16) are already in the human proteome. Their copies
there are dropped and the family FASTA's copies added, so each family protein appears
exactly once and every tool reports it under the same accession.

    stage_database.py human.fasta family.fasta database.fasta accessions.txt
"""

import sys


def records(path):
    header, seq = None, []
    for line in open(path):
        line = line.rstrip("\n")
        if line.startswith(">"):
            if header is not None:
                yield header, "".join(seq)
            header, seq = line[1:], []
        elif line:
            seq.append(line.strip())
    if header is not None:
        yield header, "".join(seq)


def accession(header):
    return header.split()[0].split("|")[1]


human, family, out, acc_out = sys.argv[1:5]
family_records = list(records(family))
family_acc = {accession(h) for h, _ in family_records}

n_human = n_dropped = 0
with open(out, "w") as fh:
    for h, s in records(human):
        if accession(h) in family_acc:
            n_dropped += 1
            continue
        n_human += 1
        fh.write(f">{h}\n{s}\n")
    for h, s in family_records:
        fh.write(f">{h}\n{s}\n")

with open(acc_out, "w") as fh:
    for h, _ in records(out):
        fh.write(accession(h) + "\n")

print(
    f"{n_human} human proteins + {len(family_records)} family proteins "
    f"({n_dropped} human copies of family members replaced) = {n_human + len(family_records)}"
)
