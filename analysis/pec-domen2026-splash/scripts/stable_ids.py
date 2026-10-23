#!/usr/bin/env python3
"""Give every FASTA record a stable unique ID (prefix + 6-digit index, in file order).

Usage: stable_ids.py <in.fasta[.gz]> <prefix> <out.faa> <map.tsv>

The original header is kept after the new ID, and map.tsv lists stable_id, original_id (first
word), the full original header, length and the sequence's sha256, so duplicate original IDs
(the new Botryllus proteome has some) never collapse. Sequences are written unchanged.
"""
import gzip
import hashlib
import sys

src, prefix, out_fa, out_map = sys.argv[1:5]
op = gzip.open if src.endswith(".gz") else open
n = 0
with op(src, "rt") as f, open(out_fa, "w") as fa, open(out_map, "w") as mp:
    mp.write("stable_id\toriginal_id\theader\tlength\tsha256\n")
    header, seq = None, []

    def flush():
        global n
        if header is None:
            return
        n += 1
        s = "".join(seq)
        sid = f"{prefix}{n:06d}"
        fa.write(f">{sid} {header}\n{s}\n")
        mp.write(f"{sid}\t{header.split()[0]}\t{header}\t{len(s)}\t{hashlib.sha256(s.encode()).hexdigest()}\n")

    for line in f:
        line = line.rstrip("\n")
        if line.startswith(">"):
            flush()
            header, seq = line[1:], []
        else:
            seq.append(line.strip())
    flush()
print(f"records {n}")
