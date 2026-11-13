"""Stream a FASTQ (plain text) from stdin and keep a fixed-seed random sample of reads.

The read count is known from ENA, so the kept read indices are drawn up front with
random.Random(seed).sample, which makes the sample reproducible in one pass. Also writes
the length of every read in the full run (for the read-length figure).
"""

import argparse

import random
import sys

p = argparse.ArgumentParser()
p.add_argument("--n-total", type=int, required=True)
p.add_argument("--n-keep", type=int, default=200_000)
p.add_argument("--seed", type=int, default=20261014)
p.add_argument("--out", required=True)
p.add_argument("--lengths", required=True)
a = p.parse_args()

keep = set(random.Random(a.seed).sample(range(a.n_total), a.n_keep))
n = 0
with open(a.out, "w") as fout, open(a.lengths, "w") as flen:
    fin = sys.stdin
    while True:
        h = fin.readline()
        if not h:
            break
        s, plus, q = fin.readline(), fin.readline(), fin.readline()
        flen.write(f"{len(s) - 1}\n")
        if n in keep:
            fout.write(h + s + "+\n" + q)
        n += 1
print(f"reads seen {n}, kept {sum(1 for i in keep if i < n)}", file=sys.stderr)
