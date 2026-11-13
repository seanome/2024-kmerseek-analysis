#!/usr/bin/env python3
"""Write each sequence's name and length from a FASTA file.

    python3 fasta_lengths.py <in.fasta> <out.tsv> [ids.txt subset.fasta]

The name is the header up to the first whitespace. Output: name<TAB>length, no header.
With ids.txt and subset.fasta, the records named in ids.txt are also written to subset.fasta
(used for BHF's own sequence on the page).
Added 2026-10-13 for notebook 281's page (Botryllus protein lengths for the pair view).
Python 3.6, standard library only.
"""
import sys


def main(path, out, ids_path=None, subset=None):
    ids = set(l.strip() for l in open(ids_path) if l.strip()) if ids_path else set()
    sub = open(subset, "w") if subset else None
    name, n = None, 0
    with open(path) as fh, open(out, "w") as w:
        for line in fh:
            if line.startswith(">"):
                if name is not None:
                    w.write("%s\t%d\n" % (name, n))
                name, n = line[1:].split()[0], 0
                keep = name in ids
                if keep:
                    sub.write(">%s\n" % name)
            else:
                n += len(line.strip())
                if keep:
                    sub.write(line.strip() + "\n")
        if name is not None:
            w.write("%s\t%d\n" % (name, n))
    if sub:
        sub.close()


if __name__ == "__main__":
    if len(sys.argv) not in (3, 5):
        sys.exit(__doc__)
    main(*sys.argv[1:])
