#!/usr/bin/env python3
"""Longest stop-free frame of each simulated transcript, to index for gene-counts (notebook 361).

Usage: control_proteins.py <control_transcripts.fa> <out.faa> <out.gene_map.tsv>

A gene is the transcript ID without its _isoA or _isoB suffix, as in control_truth.py.
"""
import re
import sys

fa, faa, gene_map = sys.argv[1:]
CODE = "KNKNTTTTRSRSIIMIQHQHPPPPRRRRLLLLEDEDAAAAGGGGVVVV*Y*YSSSS*CWCLFLF"
IDX = {"A": 0, "C": 1, "G": 2, "T": 3}
COMP = str.maketrans("ACGT", "TGCA")


def translate(s):
    return "".join(CODE[16 * IDX[s[i]] + 4 * IDX[s[i + 1]] + IDX[s[i + 2]]] if all(x in IDX for x in s[i:i + 3]) else "*"
                   for i in range(0, len(s) - 2, 3))


def records(path):
    name, seq = None, []
    for line in open(path):
        if line.startswith(">"):
            if name:
                yield name, "".join(seq).upper()
            name, seq = line[1:].split()[0], []
        else:
            seq.append(line.strip())
    yield name, "".join(seq).upper()


with open(faa, "w") as out, open(gene_map, "w") as gm:
    for name, s in records(fa):
        rc = s[::-1].translate(COMP)
        best = max((p for q in (s, rc) for off in range(3) for p in translate(q[off:]).split("*")), key=len)
        out.write(f">{name}\n{best}\n")
        gm.write(f"{name}\t{re.sub('_iso[AB]$', '', name)}\n")
