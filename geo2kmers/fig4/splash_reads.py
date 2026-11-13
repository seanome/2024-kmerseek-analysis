"""Turn one series' probe intensities into a FASTA file per sample, for SPLASH.

Usage: splash_reads.py <probes.tsv.gz> <groups.tsv> <dataset> <out folder>
SPLASH counts reads, and an array has intensities, so a probe is written 10 x intensity / the sample's
median times, at most 1000 times. A 25-letter probe has one possible anchor, its first 15 letters, so only
probes sharing an anchor with a different target are kept. Longer probes are all kept. Writes input.txt.
"""

import csv
import gzip
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ANCHOR, COPIES_AT_MEDIAN, MOST_COPIES = 15, 10, 1000
probes, groups_file, dataset, out = sys.argv[1:]
with open(groups_file) as f:
    samples = sorted({r["sample"] for r in csv.DictReader(f, delimiter="\t") if r["dataset"] == dataset})
values = defaultdict(list)
with gzip.open(probes, "rt") as f:
    header = f.readline().rstrip("\n").split("\t")
    columns = [header.index(s) for s in samples]
    for line in f:
        fields = line.rstrip("\n").split("\t")
        values[fields[0].upper()].append([float(fields[c]) for c in columns])
sequences = [s for s in values if set(s) <= set("ACGT")]
matrix = np.array([np.mean(values[s], axis=0) for s in sequences], dtype=np.float32)
copies = np.minimum(np.rint(COPIES_AT_MEDIAN * matrix / np.median(matrix, axis=0)), MOST_COPIES).astype(int)
sharing = defaultdict(set)
for s in sequences:
    sharing[s[:ANCHOR]].add(s[ANCHOR:])
keep = [i for i, s in enumerate(sequences) if len(s) > 25 or len(sharing[s[:ANCHOR]]) > 1]
Path(out).mkdir(parents=True, exist_ok=True)
with open(f"{out}/input.txt", "w") as listing:
    for j, sample in enumerate(samples):
        path = Path(out).resolve() / f"{sample}.fasta"
        listing.write(f"{sample} {path}\n")
        with open(path, "w") as f:
            for i in keep:
                f.write(f">p{i}\n{sequences[i]}\n" * copies[i, j])
print(f"{dataset}\t{len(sequences)} probes\t{len(keep)} kept")
