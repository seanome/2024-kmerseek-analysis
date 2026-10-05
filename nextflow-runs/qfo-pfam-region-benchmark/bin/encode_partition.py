#!/usr/bin/env python3
"""Rewrite a protein FASTA as a 2-letter string under a residue partition.

kmerseek's alphabets are a fixed list, so an arbitrary partition of the 20 amino acids
cannot be passed to it. Instead every residue in class 1 is written as `A` and every
residue in class 2 as `D`, and kmerseek runs on the result with `--alphabet protein20`.
An exact k-mer match on that string is the same match a kmerseek 2-letter alphabet with
the same partition makes.

The residues outside the 20 are handled the way kmerseek 0.3.1 handles them under a
reduced alphabet, so that the encoded run and a built-in alphabet with the same partition
index the same k-mers:

  U (selenocysteine), O (pyrrolysine)  take the class of C and K. kmerseek substitutes them
                                       only under a reduced alphabet, never under protein20,
                                       so it has to happen here.
  B (D or N), J (I or L), Z (E or Q)   kmerseek indexes a window holding one of these under
                                       both readings. When both readings fall in the same
                                       class that is one k-mer, and the code is written as
                                       that class. When they fall in different classes,
                                       protein20 cannot reproduce the two readings, so the
                                       code is written as X and the windows over it match
                                       nothing. The count of these is reported.
  X, *, anything else                  copied through, upper-cased, as kmerseek does.

Usage:
  encode_partition.py --partition random2_01.tsv --input in.fasta --output out.fasta
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

CANONICAL = "ACDEFGHIKLMNPQRSTVWY"
#: The letter each class is written as. Both are ordinary amino acids, so protein20
#: accepts them, and neither is an ambiguity code kmerseek would expand.
CLASS_LETTER = {1: "A", 2: "D"}
#: kmerseek 0.3.1 aminoacid.rs NONCANONICAL_AA.
NONCANONICAL = {"U": "C", "O": "K"}
#: kmerseek 0.3.1 aminoacid.rs AMBIGUITY_ALTERNATIVES.
AMBIGUITY = {"B": ("D", "N"), "J": ("I", "L"), "Z": ("E", "Q")}


def read_partition(path: Path) -> dict[str, int]:
    """Residue -> class (1 or 2) from a two-column TSV with a `residue	class` header."""
    table: dict[str, int] = {}
    with open(path) as fh:
        header = fh.readline().rstrip("\n").split("\t")
        if header != ["residue", "class"]:
            raise ValueError(
                f"{path}: header must be 'residue<TAB>class', got {header}"
            )
        for line in fh:
            if not line.strip():
                continue
            residue, cls = line.rstrip("\n").split("\t")
            table[residue] = int(cls)
    check_partition(table)
    return table


def check_partition(table: dict[str, int]) -> None:
    if sorted(table) != sorted(CANONICAL):
        raise ValueError(
            f"partition must name each of the 20 amino acids once, got {sorted(table)}"
        )
    if set(table.values()) != {1, 2}:
        raise ValueError(
            f"partition classes must be 1 and 2, got {sorted(set(table.values()))}"
        )


def translation(table: dict[str, int]) -> tuple[dict[str, str], set[str]]:
    """Per-character map for encode_sequence, and the ambiguity codes written as X."""
    out = {r: CLASS_LETTER[c] for r, c in table.items()}
    for code, analogue in NONCANONICAL.items():
        out[code] = CLASS_LETTER[table[analogue]]
    split = set()
    for code, (a, b) in AMBIGUITY.items():
        if table[a] == table[b]:
            out[code] = CLASS_LETTER[table[a]]
        else:
            out[code] = "X"
            split.add(code)
    return out, split


def encode_sequence(seq: str, mapping: dict[str, str]) -> str:
    return "".join(mapping.get(c, c) for c in seq.upper())


def _open(path: Path):
    return gzip.open(path, "rt") if str(path).endswith(".gz") else open(path)


def encode_fasta(partition: Path, fasta_in: Path, fasta_out: Path) -> dict:
    """Encode every record; headers are copied unchanged. Returns counts for the log."""
    mapping, split = translation(read_partition(partition))
    n_records = n_residues = n_split_codes = 0
    with _open(fasta_in) as fin, open(fasta_out, "w") as fout:
        for line in fin:
            if line.startswith(">"):
                n_records += 1
                fout.write(line)
                continue
            s = line.strip()
            if not s:
                continue
            n_residues += len(s)
            if split:
                n_split_codes += sum(s.upper().count(c) for c in split)
            fout.write(encode_sequence(s, mapping) + "\n")
    return {
        "partition": str(partition),
        "input": str(fasta_in),
        "n_records": n_records,
        "n_residues": n_residues,
        "ambiguity_codes_split_by_partition": sorted(split),
        "n_residues_written_as_X_for_that_reason": n_split_codes,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--partition", type=Path, required=True)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(argv)
    counts = encode_fasta(args.partition, args.input, args.output)
    print(json.dumps(counts), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
