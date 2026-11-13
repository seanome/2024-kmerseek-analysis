#!/usr/bin/env python3
"""Contig lengths of each Botryllus route, six contrasts together (notebook 361).

Usage: contig_lengths.py <dna_dir> > contig_lengths.tsv

<dna_dir> holds <contrast>.<route>.fna. Writes route, length, contigs. A contig kept in two contrasts
counts twice, as in the size table.
"""
import collections
import glob
import sys
from pathlib import Path

n = collections.Counter()
for f in glob.glob(str(Path(sys.argv[1]) / "*.fna")):
    route, length = Path(f).name.split(".")[1], None
    for line in open(f):
        if line[0] == ">":
            if length is not None:
                n[route, length] += 1
            length = 0
        else:
            length += len(line.strip())
    if length is not None:
        n[route, length] += 1
print("route\tlength\tcontigs")
for (route, length), c in sorted(n.items()):
    print(f"{route}\t{length}\t{c}")
