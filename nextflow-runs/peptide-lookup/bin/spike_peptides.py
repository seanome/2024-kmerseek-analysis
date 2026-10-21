#!/usr/bin/env python3
"""Step 1b of notebook 300: 1,000 spiked peptides with a known source protein.

Usage: spike_peptides.py <uhgp-90.faa.gz> <out.tsv> [--n 1000] [--seed 300]

Each spiked peptide is a real window of 9 to 30 residues from a UHGP-90 protein, with one or
two conservative swaps at recorded positions: I<->L, D<->E or K<->R. The protein it came from
is the truth. Two passes over the file: the first records each protein's length, the second
cuts the windows from the proteins drawn.

* Proteins are drawn with probability proportional to the number of 30-residue windows they
  hold, so every residue position in the catalogue is about equally likely to be picked.
* The window length is drawn uniformly from 9 to 30 and the window start uniformly within the
  protein.
* A window with any letter outside the 20 canonical residues, or with fewer swappable
  residues than swaps drawn, is replaced by a new draw.
* Half the peptides get one swap and half get two, at distinct positions chosen uniformly
  among the swappable residues.

Output columns: peptide_id, peptide (the query, swaps applied), source_window (before the
swaps), source_protein (UHGP-90 accession), source_start_1based, source_end_1based, length,
n_swaps, swaps (position_1based:from>to, comma-separated, positions within the peptide).
"""

import argparse
import gzip

import numpy as np

CANONICAL = set("ACDEFGHIKLMNPQRSTVWY")
SWAP = {"I": "L", "L": "I", "D": "E", "E": "D", "K": "R", "R": "K"}
L_MIN, L_MAX = 9, 30


def read_fasta(path):
    """Yield (accession, sequence) from a gzipped FASTA."""
    name, parts = None, []
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if line.startswith(">"):
                if name is not None:
                    yield name, "".join(parts)
                name, parts = line[1:].split()[0], []
            else:
                parts.append(line.strip())
    if name is not None:
        yield name, "".join(parts)


def protein_lengths(path):
    """Length of every protein, in file order, without keeping the sequences."""
    lengths, n = [], 0
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if line.startswith(">"):
                if lengths or n:
                    lengths.append(n)
                n = 0
            else:
                n += len(line.rstrip())
    lengths.append(n)
    return np.array(lengths, dtype=np.int64)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("fasta")
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=1_000)
    ap.add_argument("--seed", type=int, default=300)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    lengths = protein_lengths(args.fasta)
    weight = np.clip(lengths - L_MAX + 1, 0, None).astype(float)
    # draw extra proteins: some windows are rejected (non-canonical letters, too few swaps)
    picks = rng.choice(
        len(lengths), size=args.n * 3, replace=True, p=weight / weight.sum()
    )
    wanted = {}
    for order, idx in enumerate(picks):
        wanted.setdefault(int(idx), []).append(order)

    drawn = {}
    for idx, (name, seq) in enumerate(read_fasta(args.fasta)):
        for order in wanted.get(idx, ()):
            drawn[order] = (name, seq)

    rows = []
    for order in sorted(drawn):
        if len(rows) == args.n:
            break
        name, seq = drawn[order]
        length = int(rng.integers(L_MIN, L_MAX + 1))
        start = int(rng.integers(0, len(seq) - length + 1))
        window = seq[start : start + length]
        n_swaps = 1 if len(rows) % 2 == 0 else 2
        swappable = [i for i, aa in enumerate(window) if aa in SWAP]
        if not set(window) <= CANONICAL or len(swappable) < n_swaps:
            continue
        positions = sorted(rng.choice(swappable, size=n_swaps, replace=False).tolist())
        peptide = list(window)
        for i in positions:
            peptide[i] = SWAP[window[i]]
        rows.append(
            {
                "peptide_id": f"spike{len(rows) + 1:04d}",
                "peptide": "".join(peptide),
                "source_window": window,
                "source_protein": name,
                "source_start_1based": start + 1,
                "source_end_1based": start + length,
                "length": length,
                "n_swaps": n_swaps,
                "swaps": ",".join(
                    f"{i + 1}:{window[i]}>{SWAP[window[i]]}" for i in positions
                ),
            }
        )
    if len(rows) < args.n:
        raise RuntimeError(f"only {len(rows)} usable windows from {len(drawn)} draws")
    with open(args.out, "w") as fh:
        fh.write("\t".join(rows[0]) + "\n")
        for r in rows:
            fh.write("\t".join(str(v) for v in r.values()) + "\n")
    print(f"{len(rows)} spiked peptides from {len(lengths):_} UHGP-90 proteins")


if __name__ == "__main__":
    main()
