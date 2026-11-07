"""Write one comparison's k-mer table and design file for KaMRaT.

Usage: kamrat_table.py <probes.tsv.gz> <groups.tsv> <dataset> <comparison> <seed> <table out> <design out>
Rows are 25-letter k-mers and columns are samples, holding the probe's intensity. A k-mer on two probes
gets their mean. Longer probes give every 25-letter window. Seed 0 keeps the true groups, and any other
seed shuffles them.
"""

import csv
import gzip
import sys

import numpy as np

K = 25
probes, groups_file, dataset, comparison, seed, table_out, design_out = sys.argv[1:]
seed = int(seed)
with open(groups_file) as f:
    rows = [r for r in csv.DictReader(f, delimiter="\t") if r["dataset"] == dataset and r["comparison"] == comparison]
samples = [r["sample"] for r in rows]
labels = np.array([r["group"] for r in rows])
if seed:
    labels = np.random.default_rng(seed).permutation(labels)
with open(design_out, "w") as f:
    f.writelines(f"{s}\t{'post' if g == '1' else 'pre'}\n" for s, g in zip(samples, labels, strict=True))

totals = {}
with gzip.open(probes, "rt") as f:
    header = f.readline().rstrip("\n").split("\t")
    columns = [header.index(s) for s in samples]
    for line in f:
        fields = line.rstrip("\n").split("\t")
        sequence = fields[0].upper()
        values = np.array([float(fields[c]) for c in columns], dtype=np.float32)
        for start in range(len(sequence) - K + 1):
            kmer = sequence[start : start + K]
            if set(kmer) <= set("ACGT"):
                held = totals.get(kmer)
                totals[kmer] = (values, 1) if held is None else (held[0] + values, held[1] + 1)
with open(table_out, "w") as f:
    f.write("feature\t" + "\t".join(samples) + "\n")
    for kmer, (values, n) in totals.items():
        f.write(kmer + "\t" + "\t".join(f"{v:.2f}" for v in values / n) + "\n")
