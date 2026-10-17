#!/usr/bin/env python3
"""Summarize `kmerseek pair` JSONs: shared k-mers and the longest identical-class run.

For each ordered pair and arm:
  n_shared_kmers              k-mers present in both encoded sequences (kmerseek pair)
  n_exact_regions             exact regions they chain into (kmerseek pair)
  longest_exact_region        longest of those regions, residues; 0 when there is none
  longest_identical_class_run longest stretch, on any ungapped diagonal, where query and
                              target are in the same class at every position. It does not
                              depend on k, so it is defined even when no k-mer is shared.

    kmerseek_pair_summary.py alphabet ksize out.tsv pair_*.json
"""

import json
import sys

import polars as pl

alphabet, ksize, out, *paths = sys.argv[1:]


def longest_run(a, b):
    best = 0
    for d in range(-(len(a) - 1), len(b)):
        run = 0
        for i in range(max(0, -d), min(len(a), len(b) - d)):
            if a[i] == b[i + d]:
                run += 1
                best = max(best, run)
            else:
                run = 0
    return best


rows = []
for p in paths:
    d = json.load(open(p))
    acc = lambda s: s["name"].split()[0].split("|")[1]
    rows.append(
        dict(
            query=acc(d["query"]),
            target=acc(d["target"]),
            alphabet=alphabet,
            ksize=int(ksize),
            n_shared_kmers=len(d["shared_kmers"]),
            n_exact_regions=len(d["regions"]),
            longest_exact_region=max((r["length"] for r in d["regions"]), default=0),
            longest_identical_class_run=longest_run(
                d["query"]["encoded"], d["target"]["encoded"]
            ),
        )
    )
pl.DataFrame(rows).write_csv(out, separator="\t")
print(f"{alphabet} k{ksize}: {len(rows)} pairs")
