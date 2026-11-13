#!/usr/bin/env python3
"""Share of each protein's residues that SEG marks as low-complexity, for notebook 344.

    seg_fraction.py <proteins.fasta> <segmasker_intervals.txt> <out.tsv>

<segmasker_intervals.txt> is `segmasker -outfmt interval` output with segmasker's default
settings (window 12, low cut 2.2, high cut 2.5; Wootton and Federhen 1993): a `>` header
line per protein, then one `start - end` line per masked stretch, both ends included.

Writes one row per protein in FASTA order: id (first word of the header), length,
seg_residues (masked residues), seg_fraction (seg_residues / length).
Standard library only.
"""

import csv
import sys


def read_lengths(path):
    lengths, name = {}, None
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith(">"):
                name = line[1:].split()[0]
                if name in lengths:
                    sys.exit(f"{name} appears twice in {path}")
                lengths[name] = 0
            elif line:
                lengths[name] += len(line)
    return lengths


def read_masked(path, names):
    masked = {n: set() for n in names}
    name = None
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith(">"):
                name = line[1:].split()[0]
                if name not in masked:
                    sys.exit(f"segmasker reports {name}, which is not in the FASTA")
            elif line:
                start, end = (int(x) for x in line.split(" - "))
                masked[name].update(range(start, end + 1))
    return masked


def main(fasta, intervals, out):
    lengths = read_lengths(fasta)
    masked = read_masked(intervals, lengths)
    with open(out, "w") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(["id", "length", "seg_residues", "seg_fraction"])
        for name, length in lengths.items():
            n = len(masked[name])
            if n > length or (masked[name] and max(masked[name]) >= length):
                sys.exit(f"{name}: masked positions reach past its {length} residues")
            w.writerow([name, length, n, f"{n / length:.6f}" if length else "nan"])


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    main(*sys.argv[1:])
