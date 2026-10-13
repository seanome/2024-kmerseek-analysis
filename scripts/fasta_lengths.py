#!/usr/bin/env python3
"""Write each sequence's name and length from a FASTA file.

    python3 fasta_lengths.py <in.fasta> <out.tsv>

The name is the header up to the first whitespace. Output: name<TAB>length, no header.
Added 2026-10-13 for notebook 281's page (Botryllus protein lengths for the pair view).
Python 3.6, standard library only.
"""
import sys


def main(path, out):
    name, n = None, 0
    with open(path) as fh, open(out, "w") as w:
        for line in fh:
            if line.startswith(">"):
                if name is not None:
                    w.write("%s\t%d\n" % (name, n))
                name, n = line[1:].split()[0], 0
            else:
                n += len(line.strip())
        if name is not None:
            w.write("%s\t%d\n" % (name, n))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(*sys.argv[1:])
