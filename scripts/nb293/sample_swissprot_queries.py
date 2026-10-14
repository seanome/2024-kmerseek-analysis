"""Write a random sample of Swiss-Prot proteins as a FASTA, for timing step 6 on a subset.

Usage: python3 -I sample_swissprot_queries.py SWISSPROT_FASTA_GZ N SEED OUT_FASTA
"""

import gzip
import random
import sys

path, n, seed, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
records, header, seq = [], None, []
with gzip.open(path, "rt") as fh:
    for line in fh:
        if line.startswith(">"):
            if header:
                records.append((header, "".join(seq)))
            header, seq = line.rstrip("\n"), []
        else:
            seq.append(line.strip())
records.append((header, "".join(seq)))
records.sort()
with open(out, "w") as fh:
    for h, s in random.Random(seed).sample(records, n):
        fh.write(f"{h}\n{s}\n")
print(f"{n} of {len(records)} proteins -> {out}")
