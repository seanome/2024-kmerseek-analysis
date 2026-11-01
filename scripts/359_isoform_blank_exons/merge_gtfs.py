#!/usr/bin/env python3
"""Notebook 359: join ENCODE's per-replicate TALON GTFs into one transcript set.

    merge_gtfs.py --genome-fai-from <genome.fa> --out <union.gtf> <rep1.gtf.gz> ...

TALON gives a transcript the same ID in every replicate of one experiment, so the union
is taken by transcript_id. A transcript seen in several replicates must have the same
exons in each; the script stops if not. Only transcript and exon lines are kept, and
only on chromosomes in the genome FASTA (spike-in sequences are dropped and counted).
Standard library only.
"""

import argparse
import gzip
import re
import sys


def chroms(fasta):
    out = set()
    with open(fasta) as fh:
        for line in fh:
            if line.startswith(">"):
                out.add(line[1:].split()[0])
    return out


def main(a):
    keep_chroms = chroms(a.genome_fai_from)
    tx, order, dropped = {}, [], {}
    for path in a.gtfs:
        seen = {}
        with gzip.open(path, "rt") as fh:
            for line in fh:
                if line.startswith("#"):
                    continue
                f = line.rstrip("\n").split("\t")
                if len(f) < 9 or f[2] not in ("transcript", "exon"):
                    continue
                if f[0] not in keep_chroms:
                    dropped[f[0]] = dropped.get(f[0], 0) + 1
                    continue
                tid = re.search(r'transcript_id "([^"]+)"', f[8]).group(1)
                seen.setdefault(tid, []).append(f)
        for tid, lines in seen.items():
            exons = sorted((l[0], l[6], int(l[3]), int(l[4])) for l in lines if l[2] == "exon")
            if tid in tx:
                if tx[tid][0] != exons:
                    sys.exit(f"{tid}: exons differ between replicates")
            else:
                tx[tid] = (exons, lines)
                order.append(tid)
        print(f"{path}: {len(seen)} transcripts", file=sys.stderr)
    with open(a.out, "w") as fo:
        for tid in order:
            for f in tx[tid][1]:
                fo.write("\t".join(f) + "\n")
    print(f"union: {len(order)} transcripts; lines dropped off-genome: {dropped}", file=sys.stderr)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--genome-fai-from", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("gtfs", nargs="+")
    main(p.parse_args())
