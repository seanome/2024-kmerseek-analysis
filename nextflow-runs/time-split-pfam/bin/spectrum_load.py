#!/usr/bin/env python3
"""Spectrum load of a reference at a few kmerseek settings (notebook 280, Stop A sizing).

    spectrum_load.py --fasta <reference.fasta> [--fasta ...] --out <csv>

Spectrum load is sum(occ^2 n) / sum(occ n) over a reference's k-mer spectrum, where n
k-mers occur in occ sequences each (a k-mer repeated within one sequence counts once, as
kmerseek counts it): the dark-set model of kmerseek search memory,
peak ~ 36 + 2.34 sqrt(load) GB (main.nf, searchFirstGb). kmerseekIndex writes the same
spectrum as spectrum.csv.gz; a test setting can be checked against it with --check.

Alphabets here are the ones whose residue classes are written out in
notebooks/ortholog_analysis_utils.py: protein20 (the residues themselves) and
hp_pbotc_1st_ed2 (h = ACFILMPVWY, p = DEGHKNQRST). A k-mer holding any other letter is
not counted.

Also writes <out>_lengths.csv: reference, entry length (aa), number of entries.
"""

from __future__ import annotations

import argparse
import gzip
from pathlib import Path

import numpy as np

ALPHABETS = {
    "protein20": ["A", "C", "D", "E", "F", "G", "H", "I", "K", "L",
                  "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"],
    "hp_pbotc_1st_ed2": ["ACFILMPVWY", "DEGHKNQRST"],
}
SETTINGS = [("protein20", 5), ("protein20", 8), ("hp_pbotc_1st_ed2", 17), ("hp_pbotc_1st_ed2", 29)]


def residues(path: Path) -> np.ndarray:
    """Every distinct sequence once, as bytes, each followed by one '*' so no k-mer spans two
    entries. Distinct, because the kmerseek 0.4.0 index holds identical sequences as one
    entry (it merges entries with the same k-mer set; kmerseek PR 141), and its spectrum
    counts that one entry. load() also merges sequences that are equal in the alphabet."""
    parts, buf = [], []
    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                if buf:
                    parts.append("".join(buf) + "*")
                buf = []
            else:
                buf.append(line.strip())
    if buf:
        parts.append("".join(buf) + "*")
    return np.frombuffer("".join(dict.fromkeys(parts)).encode(), dtype=np.uint8)


def entry_lengths(path: Path) -> np.ndarray:
    """Number of entries of each length (aa), every entry of the file counted."""
    lens, n = [], None
    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                if n is not None:
                    lens.append(n)
                n = 0
            else:
                n += len(line.strip())
    if n is not None:
        lens.append(n)
    return np.bincount(np.asarray(lens))


def load(seq: np.ndarray, classes: list[str], k: int) -> tuple[float, int, int]:
    lut = np.full(256, -1, dtype=np.int64)
    for i, letters in enumerate(classes):
        for c in letters:
            lut[ord(c)] = i
    # Sequences that are the same in this alphabet are one index entry too (the merge is on
    # the k-mer set, which the alphabet decides), so dedupe the encoded sequences.
    enc = np.where(lut[seq] < 0, 255, lut[seq]).astype(np.uint8)
    enc[seq == ord("*")] = 254
    parts = bytes(enc).split(bytes([254]))
    enc = np.frombuffer(b"".join(p + bytes([254]) for p in dict.fromkeys(x for x in parts if x)), dtype=np.uint8)
    x = enc.astype(np.int64)
    x[x >= 254] = -1
    seq = np.where(enc == 254, ord("*"), 0).astype(np.uint8)
    bad = np.concatenate([[0], np.cumsum(x < 0)])
    ok = (bad[k:] - bad[:-k]) == 0  # window i..i+k-1 holds no unknown letter
    base = len(classes)
    code = np.zeros(len(x) - k + 1, dtype=np.int64)
    xc = np.where(x < 0, 0, x)
    for j in range(k):
        code = code * base + xc[j : len(x) - k + 1 + j]
    # kmerseek's spectrum counts the sequences holding a k-mer, not its positions: one
    # k-mer repeated inside a sequence counts once (spectrum.csv.gz "seqs_per_kmer").
    seq_id = np.cumsum(seq == ord("*"))[: len(code)]
    assert code.max() < 2**42 and seq_id.max() < 2**21, "packed key would overflow"
    pairs = np.unique((code[ok] << 21) | seq_id[ok])
    _, occ = np.unique(pairs >> 21, return_counts=True)
    occ = occ.astype(np.float64)
    return float((occ**2).sum() / occ.sum()), int(occ.sum()), len(occ)


def spectrum_file_load(path: Path) -> float:
    num = den = 0.0
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if line.startswith("#") or line.startswith("moltype"):
                continue
            c = line.rstrip("\n").split(",")
            occ, n = float(c[2]), float(c[3])
            num += occ * occ * n
            den += occ * n
    return num / den


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--check", nargs=4, metavar=("FASTA", "ALPHABET", "K", "SPECTRUM_CSV_GZ"),
                    help="compare this script's load with a kmerseek index's spectrum file")
    a = ap.parse_args()
    rows = ["reference,alphabet,ksize,load,kmers,distinct_kmers,predicted_search_gb"]
    lengths = ["reference,length,entries"]
    for f in a.fasta:
        seq = residues(f)
        n_len = entry_lengths(f)
        lengths += [f"{f.parent.name},{L},{n}" for L, n in enumerate(n_len) if n]
        for alph, k in SETTINGS:
            ld, n, d = load(seq, ALPHABETS[alph], k)
            rows.append(f"{f.parent.name},{alph},{k},{ld:.2f},{n},{d},{36 + 2.34 * ld ** 0.5:.1f}")
            print(rows[-1], flush=True)
    if a.check:
        f, alph, k, spec = a.check
        mine = load(residues(Path(f)), ALPHABETS[alph], int(k))[0]
        theirs = spectrum_file_load(Path(spec))
        rows.append(f"check {Path(f).parent.name},{alph},{k},{mine:.2f} (this script) vs {theirs:.2f} (kmerseek),,,")
        print(rows[-1])
    a.out.write_text("\n".join(rows) + "\n")
    a.out.with_name(a.out.stem + "_lengths.csv").write_text("\n".join(lengths) + "\n")


if __name__ == "__main__":
    main()
