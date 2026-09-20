#!/usr/bin/env python3
"""The dark proteins with their residues shuffled: a null for kmerseek reach.

Each dark protein keeps its length and its amino-acid composition and loses everything
else, so any region kmerseek reports on the shuffled copy is reached by composition alone.
The shuffled copies are searched with the same alphabet, k and mask settings as the real
proteins, and the report shows the two reaches side by side: a dark reach the shuffled
sequences match is not a homology signal.

Accessions are kept verbatim so the reach can be counted against the same dark set. The
output is split into chunks of the same size the real proteome was, so each search is one
job of the same shape.
"""
import argparse
import random
from pathlib import Path

import polars as pl


def read_fasta(path: Path):
    acc, seq = None, []
    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                if acc is not None:
                    yield acc, "".join(seq)
                acc, seq = line[1:].strip().split()[0], []
            else:
                seq.append(line.strip())
    if acc is not None:
        yield acc, "".join(seq)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--query", type=Path, required=True,
                    help="the proteome every arm searched, bare-accession headers")
    ap.add_argument("--dark", type=Path, required=True, help="<species>_dark_set.parquet")
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--chunk-size", type=int, required=True)
    ap.add_argument("--seed", type=int, default=20260920)
    args = ap.parse_args()

    dark = set(pl.read_parquet(args.dark)["accession"].to_list())
    rng = random.Random(args.seed)
    args.outdir.mkdir(parents=True, exist_ok=True)
    n_written = 0
    chunk_i = 0
    fh = None
    for acc, seq in read_fasta(args.query):
        if acc not in dark:
            continue
        if n_written % args.chunk_size == 0:
            if fh:
                fh.close()
            fh = open(args.outdir / f"chunk_{chunk_i:04d}.fasta", "w")
            chunk_i += 1
        residues = list(seq)
        rng.shuffle(residues)
        fh.write(f">{acc}\n{''.join(residues)}\n")
        n_written += 1
    if fh:
        fh.close()
    missing = len(dark) - n_written
    if missing:
        raise SystemExit(f"{missing} dark accessions are not in {args.query}: the dark "
                         f"set and the proteome disagree on the key.")
    print(f"shuffled {n_written} dark proteins into {chunk_i} chunk(s) of "
          f"{args.chunk_size} under {args.outdir}")


if __name__ == "__main__":
    main()
