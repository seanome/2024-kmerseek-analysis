"""Write platforms.tsv: each of the eight series' platform, whether it is an array, and its landing share.

Usage: platforms.py <landing share files ...> <out.tsv>
A series with no landing share file is RNA-seq, and geo2kmers does not cover it.
"""

import csv
import sys

RNA_SEQ = {"GSE120862": "GPL16791,GPL18573", "GSE99963": "GPL16791", "GSE144304": "GPL18573"}
*shares, out = sys.argv[1:]
rows = []
for path in shares:
    with open(path) as f:
        rows += list(csv.DictReader(f, delimiter="\t"))
columns = ["gse", "gpl", "type", "median_probe_length", "rule", "k", "probes", "landing_share", "features_landed"]
columns += ["proteins_landed", "orientation", "probes_with_a_kmer_as_given", "probes_with_a_kmer_complemented"]
with open(out, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=columns, delimiter="\t", lineterminator="\n", extrasaction="ignore")
    writer.writeheader()
    for row in sorted(rows, key=lambda r: r["gse"]):
        writer.writerow({**row, "type": "array"})
    for gse, gpl in RNA_SEQ.items():
        writer.writerow({"gse": gse, "gpl": gpl, "type": "RNA-seq, not covered by geo2kmers"})
