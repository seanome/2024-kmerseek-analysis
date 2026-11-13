#!/usr/bin/env python3
"""Decoy queries for notebook 344: shuffled copies of every protein in a FASTA.

    make_decoys.py <proteins.fasta> <replicates> <chunk_size> <out_dir>

A decoy is a real protein with its residues reordered, so any region kmerseek finds for it
is a chance match. Three kinds, from weakest to strongest null:

  composition  residues in a random order; keeps only the count of each amino acid
  dipeptide    keeps the first and last residue and the count of every adjacent residue
               pair (Altschul and Erickson 1985); the null the release gate is judged on
  window10     residues shuffled inside each non-overlapping 10-residue stretch, so each
               stretch keeps its hydrophobicity and long low-complexity runs survive

The shuffles are the ones in nextflow-runs/shared/decoys.py, imported from PYTHONPATH.
Each copy's random generator is seeded by sha256("344:<kind>:<replicate>:<accession>"), so
a decoy does not depend on file order or chunking. Every copy is checked against what its
kind must keep.

Writes, in <out_dir>:
  <kind>.chunk_NNNN.fasta  at most <chunk_size> decoys of one kind, named
                           <accession>__<kind>_r<replicate>
  decoys.tsv               decoy_id, source_id, accession, kind, replicate, length,
                           unchanged (1 when the shuffle returned the protein itself),
                           same_hp_lehninger2 (1 when the decoy's hydrophobic/polar
                           string under hp_lehninger2 equals the source's: searched in
                           that alphabet, it is the protein itself, not a decoy)
Standard library only.
"""

import csv
import hashlib
import os
import random
import sys
from collections import Counter

from decoys import composition_shuffle, dipeptide_shuffle, window_shuffle

SEED_LABEL = 344
WINDOW = 10
KINDS = ("composition", "dipeptide", "window10")
# hp_lehninger2's classes, from kmerseek src/rust/alphabets.rs (LEHNINGER_HP).
HP_LEHNINGER2 = str.maketrans("AFGILMPVWYCDEHKNQRST", "hhhhhhhhhhpppppppppp")


def read_fasta(path):
    """(first word of the header, sequence) in file order."""
    records, name, buf = [], None, []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith(">"):
                if name is not None:
                    records.append((name, "".join(buf)))
                name, buf = line[1:].split()[0], []
            elif line:
                buf.append(line)
    if name is not None:
        records.append((name, "".join(buf)))
    return records


def accession(name):
    """P10415 from sp|P10415|BCL2_HUMAN; the name itself when it has no | fields."""
    parts = name.split("|")
    return parts[1] if len(parts) >= 3 else name


def rng_for(kind, replicate, acc):
    h = hashlib.sha256(f"{SEED_LABEL}:{kind}:{replicate}:{acc}".encode()).digest()
    return random.Random(int.from_bytes(h[:8], "big"))


def pair_counts(seq):
    return Counter(zip(seq, seq[1:]))


def shuffle_one(kind, seq, rng):
    """The decoy, after checking it keeps what its kind must keep."""
    if kind == "composition":
        out = composition_shuffle(seq, rng)
    elif kind == "dipeptide":
        out = dipeptide_shuffle(seq, rng)
        if (
            pair_counts(out) != pair_counts(seq)
            or out[:1] != seq[:1]
            or out[-1:] != seq[-1:]
        ):
            raise ValueError("dipeptide shuffle changed a pair count or an end residue")
    elif kind == "window10":
        out = window_shuffle(seq, rng, WINDOW)
        for i in range(0, len(seq), WINDOW):
            if Counter(out[i : i + WINDOW]) != Counter(seq[i : i + WINDOW]):
                raise ValueError(
                    f"window shuffle changed the residues of the window at {i}"
                )
    else:
        raise ValueError(f"unknown kind {kind}")
    if Counter(out) != Counter(seq):
        raise ValueError(f"{kind} shuffle changed the composition")
    return out


def main(fasta, replicates, chunk_size, out_dir):
    replicates, chunk_size = int(replicates), int(chunk_size)
    records = read_fasta(fasta)
    accs = [accession(n) for n, _ in records]
    dup = [a for a, c in Counter(accs).items() if c > 1]
    if dup:
        sys.exit(
            f"accessions appear more than once, decoy names would collide: {dup[:5]}"
        )
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for kind in KINDS:
        decoys = []
        for (name, seq), acc in zip(records, accs):
            for r in range(1, replicates + 1):
                try:
                    out = shuffle_one(kind, seq, rng_for(kind, r, acc))
                except ValueError as e:
                    sys.exit(f"{name}: {e}")
                decoy_id = f"{acc}__{kind}_r{r}"
                decoys.append((decoy_id, out))
                same_hp = out.translate(HP_LEHNINGER2) == seq.translate(HP_LEHNINGER2)
                rows.append(
                    [
                        decoy_id,
                        name,
                        acc,
                        kind,
                        r,
                        len(seq),
                        int(out == seq),
                        int(same_hp),
                    ]
                )
        for c, start in enumerate(range(0, len(decoys), chunk_size)):
            path = os.path.join(out_dir, f"{kind}.chunk_{c:04d}.fasta")
            with open(path, "w") as fh:
                for decoy_id, seq in decoys[start : start + chunk_size]:
                    fh.write(f">{decoy_id}\n{seq}\n")
        print(
            f"{kind}: {len(decoys)} decoys from {len(records)} proteins, "
            f"{sum(r[6] for r in rows if r[3] == kind)} unchanged, "
            f"{sum(r[7] for r in rows if r[3] == kind)} with the source's hp_lehninger2 string"
        )
    with open(os.path.join(out_dir, "decoys.tsv"), "w") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(
            [
                "decoy_id",
                "source_id",
                "accession",
                "kind",
                "replicate",
                "length",
                "unchanged",
                "same_hp_lehninger2",
            ]
        )
        w.writerows(rows)


if __name__ == "__main__":
    if len(sys.argv) != 5:
        sys.exit(__doc__)
    main(*sys.argv[1:])
