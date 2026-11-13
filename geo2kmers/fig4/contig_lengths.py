"""Count how long KaMRaT's merged contigs are, and how many 25-letter k-mers went into each.

Usage: contig_lengths.py <out.tsv> <name>.kamrat_contigs.tsv ...
KaMRaT merge joins the 10000 top k-mers of each comparison when they overlap by 13 to 24 letters
(fig4_splash_kamrat.nf). Writes one row per comparison and contig length, with the number of contigs.
"""

import csv
import sys
from collections import Counter

out, *tables = sys.argv[1:]
with open(out, "w") as f:
    f.write("comparison\tcontig_length\tcontigs\tkmers_merged\n")
    for table in tables:
        name = table.split("/")[-1].removesuffix(".0.kamrat_contigs.tsv")
        lengths, merged = Counter(), Counter()
        with open(table) as g:
            for row in csv.DictReader(g, delimiter="\t"):
                lengths[len(row["contig"])] += 1
                merged[len(row["contig"])] += int(row["nb-merged-kmer"])
        for length in sorted(lengths):
            f.write(f"{name}\t{length}\t{lengths[length]}\t{merged[length]}\n")
