#!/usr/bin/env python3
"""Stop codons in each of a read's six reading frames, and the longest stop-free stretch.

Usage: frame_stops.py < reads.fastq > stops.tsv

One row per read: read_id, read_len, then per frame the number of stop codons (stops0-5) and the
longest run of codons with no stop (run0-5, in codons). Frames 0-2 start at offsets 0-2 of the
read, frames 3-5 at offsets 0-2 of its reverse complement (the order of pec-tools frame-scores).
A stop codon is TAA, TAG or TGA; a codon with any other letter (N) is not a stop. This is the
whole of the "stop codons alone" method: it uses no reference, so its run time is the method's.
"""
import re
import sys

STOP = re.compile(r"(?=(?:TAA|TAG|TGA))")
RC = str.maketrans("ACGTN", "TGCAN")


def frame_stats(s):
    """(stops, longest stop-free run in codons) for offsets 0, 1, 2 of one strand."""
    pos = [m.start() for m in STOP.finditer(s)]
    out = []
    for off in range(3):
        n = (len(s) - off) // 3
        cod = [(p - off) // 3 for p in pos if p % 3 == off and p + 3 <= len(s)]
        if not cod:
            out.append((0, n))
            continue
        runs = [cod[0]] + [b - a - 1 for a, b in zip(cod, cod[1:])] + [n - 1 - cod[-1]]
        out.append((len(cod), max(runs)))
    return out


def main():
    w = sys.stdout.write
    w("read_id\tread_len\t" + "\t".join(f"stops{i}" for i in range(6)) + "\t"
      + "\t".join(f"run{i}" for i in range(6)) + "\n")
    lines = iter(sys.stdin)
    for h in lines:
        if not h.strip():
            continue
        seq = next(lines).strip().upper()
        next(lines)
        next(lines)
        st = frame_stats(seq) + frame_stats(seq.translate(RC)[::-1])
        w(h[1:].split()[0] + f"\t{len(seq)}\t" + "\t".join(str(a) for a, _ in st) + "\t"
          + "\t".join(str(b) for _, b in st) + "\n")


if __name__ == "__main__":
    main()
