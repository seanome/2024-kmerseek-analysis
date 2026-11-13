"""Count the k-mers KaMRaT's t-test finds, and keep the 1000 best.

Usage: kamrat_summary.py <scores.tsv> <dataset> <comparison> <seed> <summary out> <top out>
Reads KaMRaT's table, whose first column is the k-mer and second its adjusted p value.
"""

import csv
import heapq
import sys

FDR = 0.05
scores, dataset, comparison, seed, summary_out, top_out = sys.argv[1:]
tested = passing = 0
best = []
with open(scores) as f:
    next(f)
    for line in f:
        kmer, value = line.split("\t", 2)[:2]
        q = float(value)
        tested += 1
        passing += q < FDR
        heapq.heappush(best, (-q, kmer))
        if len(best) > 1000:
            heapq.heappop(best)
with open(summary_out, "w", newline="") as f:
    writer = csv.writer(f, delimiter="\t", lineterminator="\n")
    writer.writerow(["dataset", "comparison", "seed", "kmers_tested", "kmers_below_fdr"])
    writer.writerow([dataset, comparison, seed, tested, passing])
with open(top_out, "w") as f:
    f.write("kmer\tq\n")
    f.writelines(f"{kmer}\t{-q:.4g}\n" for q, kmer in sorted(best, reverse=True))
