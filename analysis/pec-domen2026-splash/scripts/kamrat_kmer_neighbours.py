#!/usr/bin/env python3
"""Why KaMRaT k-mers stay unmerged: do they have a next k-mer in the table? (notebook 361)

Usage: kamrat_kmer_neighbours.py <raw-counts.tsv.gz> <contigs_all.tsv.gz> <out.tsv>

For every contig of one k-mer (31 bases), looks up the four possible k-mers one base further on each
side, in the k-mer table KaMRaT indexed, on either strand. Each side is "none" (table has no next
k-mer, so the merge has nothing to join), "one" or "two or more" (a fork). Also gives the highest
count of each table k-mer in any sample, to show how close these k-mers sit to DE-kupl's cutoff of 5
reads in 2 samples. Writes side counts, and prints them.
"""
import gzip
import sys

import pandas as pd

counts_gz, contigs_gz, out_tsv = sys.argv[1:]
COMP = str.maketrans("ACGT", "TGCA")


def canon(s):
    r = s.translate(COMP)[::-1]
    return min(s, r)


kmers = {}
with gzip.open(counts_gz, "rt") as fh:
    next(fh)
    for line in fh:
        f = line.rstrip("\n").split("\t")
        kmers[canon(f[0])] = max(int(float(x)) for x in f[1:])
single = pd.read_csv(contigs_gz, sep="\t", usecols=[0, 1])
single.columns = ["contig", "n"]
long_kmers = sum(len(c) - 30 for c in single.contig[single.n > 1])
single = single.contig[single.n == 1]
rows = []
for s in single:
    left = sum(canon(b + s[:30]) in kmers for b in "ACGT")
    right = sum(canon(s[1:] + b) in kmers for b in "ACGT")
    rows.append(dict(left=min(left, 2), right=min(right, 2), top=kmers.get(canon(s))))
d = pd.DataFrame(rows)
NAME = {0: "none", 1: "one", 2: "two or more"}
t = d.groupby(["left", "right"]).agg(contigs=("top", "size"), median_top_count=("top", "median")).reset_index()
t["left"], t["right"] = t.left.map(NAME), t.right.map(NAME)
t["pct"] = (100 * t.contigs / t.contigs.sum()).round(1)
t.to_csv(out_tsv, sep="\t", index=False)
print("table k-mers", len(kmers), "one-k-mer contigs", len(single), "k-mers in longer contigs", long_kmers)
print(t.to_string(index=False))
print("median top count: one-k-mer contigs", d.top.median(), "; all table k-mers", pd.Series(list(kmers.values())).median())
