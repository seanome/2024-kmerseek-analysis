#!/usr/bin/env python3
"""A classifier that runs no search: each query window's nearest regions-index entry by
amino-acid composition.

Each query is cut into --window-aa residue windows every --step residues (the last window
ends at the protein's last residue; a protein shorter than one window is one window). Each
window and each index entry (targets and decoys) is described by its 20 amino-acid
frequencies; letters outside the 20 are ignored. A window is labelled with the entry at the
smallest Euclidean distance, and is written as one call whose score is that distance, in
the shared call format (score_calls.py ranks calls by the `evalue` column, smallest first):

  query, qstart, qend, target, tstart, tend, evalue, qseq, tseq

So the classifier is held to the same 5% decoy rule and the same correctness rule as the
search tools, and a decoy entry wins a window whenever its composition is closer.

A window-shuffled decoy has exactly its source entry's composition, so every target ties
with its own decoy. Ties are broken at random (seeded), not by file order: file order puts
targets first and would hide every decoy (931 of 931 windows went to targets in the
2026-10-07 test). The number of entries tied at the smallest distance is written per call.
"""

import argparse
import gzip
import sys
from pathlib import Path

import numpy as np

AA = "ACDEFGHIKLMNPQRSTVWY"
IDX = {a: i for i, a in enumerate(AA)}


def read_fasta(path: Path):
    name, seq = None, []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if name is not None:
                    yield name, "".join(seq)
                name, seq = line[1:].split()[0], []
            elif line:
                seq.append(line.strip())
    if name is not None:
        yield name, "".join(seq)


def freqs(seq: str) -> np.ndarray:
    v = np.zeros(20, dtype=np.float32)
    for c in seq:
        i = IDX.get(c)
        if i is not None:
            v[i] += 1
    s = v.sum()
    return v / s if s else v


def windows(seq: str, size: int, step: int):
    n = len(seq)
    if n <= size:
        yield 1, n
        return
    starts = list(range(0, n - size + 1, step))
    if starts[-1] + size < n:
        starts.append(n - size)
    for s in starts:
        yield s + 1, s + size


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--index", type=Path, required=True, help="target + decoy FASTA")
    ap.add_argument("--window-aa", type=int, default=30)
    ap.add_argument("--step", type=int, default=10)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--seed", type=int, default=20261007)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    names, seqs, vecs = [], [], []
    for name, seq in read_fasta(args.index):
        names.append(name)
        seqs.append(seq)
        vecs.append(freqs(seq))
    E = np.vstack(vecs)
    e2 = (E * E).sum(axis=1)

    wins = []
    for q, s in read_fasta(args.queries):
        for a, b in windows(s, args.window_aa, args.step):
            wins.append((q, a, b, s[a - 1 : b]))
    W = np.vstack([freqs(w[3]) for w in wins])
    rng = np.random.default_rng(args.seed)

    with gzip.open(args.out, "wt") as fh:
        fh.write(
            "query\tqstart\tqend\ttarget\ttstart\ttend\tevalue\tqseq\ttseq\tn_tied\n"
        )
        for i in range(0, len(wins), args.chunk):
            w = W[i : i + args.chunk]
            d2 = (w * w).sum(axis=1)[:, None] + e2[None, :] - 2.0 * (w @ E.T)
            dmin = d2.min(axis=1)
            picks, ties = [], []
            for r in range(len(d2)):
                tied = np.flatnonzero(d2[r] <= dmin[r] + 1e-9)
                picks.append(tied[rng.integers(len(tied))])
                ties.append(len(tied))
            dist = np.sqrt(np.maximum(dmin, 0.0))
            for k, (jj, dd) in enumerate(zip(picks, dist)):
                q, a, b, wseq = wins[i + k]
                fh.write(
                    f"{q}\t{a}\t{b}\t{names[jj]}\t1\t{len(seqs[jj])}\t{dd:.6g}"
                    f"\t{wseq}\t{seqs[jj]}\t{ties[k]}\n"
                )
    print(
        f"[composition] windows={len(wins)} entries={len(names)} "
        f"window={args.window_aa} step={args.step}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
