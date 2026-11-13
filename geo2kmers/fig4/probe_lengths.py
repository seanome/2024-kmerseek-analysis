"""Count how long each series' probes are, and so how many 25-letter k-mers each probe gives.

Usage: probe_lengths.py <out.tsv> <GSE>.probes.tsv.gz ...
KaMRaT's k-mers are every 25-letter window of a probe (kamrat_table.py), so a 25-letter probe gives one k-mer
and a 60-letter probe gives 36. Writes one row per series and probe length, with the number of probes.
"""

import gzip
import sys
from collections import Counter

K = 25
out, *tables = sys.argv[1:]
with open(out, "w") as f:
    f.write("gse\tprobe_length\tprobes\tkmers_per_probe\n")
    for table in tables:
        gse = table.split("/")[-1].split(".")[0]
        with gzip.open(table, "rt") as g:
            g.readline()
            lengths = Counter(len(line.split("\t", 1)[0]) for line in g)
        for length, n in sorted(lengths.items()):
            f.write(f"{gse}\t{length}\t{n}\t{max(0, length - K + 1)}\n")
