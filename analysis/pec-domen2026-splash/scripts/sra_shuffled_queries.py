#!/usr/bin/env python3
"""Shuffle the residues of each list protein, to show what chance alone matches (notebook 361).

Usage: sra_shuffled_queries.py <queries.faa> <query_map.tsv> <out.faa>

One shuffled copy of each protein with set = paper_list, named shuf_<protein>. Length and residue
content stay the same, so homology is the only thing removed. Seed 361.
"""
import csv
import random
import sys

faa, qmap, out = sys.argv[1:]
keep = {r["query"] for r in csv.DictReader(open(qmap), delimiter="\t") if r["set"] == "paper_list"}
rng = random.Random(361)
seqs, name = {}, None
for line in open(faa):
    if line[0] == ">":
        name = line[1:].split()[0]
        seqs[name] = []
    else:
        seqs[name].append(line.strip())
with open(out, "w") as f:
    for name in sorted(keep):
        s = list("".join(seqs[name]))
        rng.shuffle(s)
        f.write(f">shuf_{name}\n{''.join(s)}\n")
print("shuffled proteins", len(keep))
