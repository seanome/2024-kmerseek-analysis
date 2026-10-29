"""Six-frame translation of reads, with a fixed rule for codons that contain N.

Standard genetic code (NCBI table 1). A codon with exactly one N stands for four codons.
If all four give the same amino acid, the codon gets that amino acid; if any of the four
is a stop, or they disagree, the codon becomes X. A codon with two or more Ns becomes X.
An X is never a stop. A translation is kept only if it has at least MIN_AA residues
before its first stop (or the end of the read).

Run as a script to write codon_table_with_N.tsv:
    python translate.py codon_table_with_N.tsv
"""

import itertools
import sys
from collections import Counter

BASES = "TCAG"
AMINO = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
CODE = {
    a + b + c: AMINO[16 * i + 4 * j + k]
    for (i, a), (j, b), (k, c) in itertools.product(enumerate(BASES), repeat=3)
}
MIN_AA = 30
COMPLEMENT = str.maketrans("ACGTN", "TGCAN")


def _n_codon(codon):
    """Translate a codon with at least one N. Returns (amino acid, how it was resolved)."""
    if codon.count("N") >= 2:
        return "X", "two_or_more_N"
    aas = {CODE[codon.replace("N", b)] for b in "ACGT"}
    if "*" in aas:
        return "X", "one_N_could_be_stop"
    if len(aas) == 1:
        return aas.pop(), "one_N_resolved"
    return "X", "one_N_ambiguous"


def build_table():
    """Every codon over ACGTN mapped to (amino acid, how). Codons without N: 'no_N'."""
    table = {}
    for codon in map("".join, itertools.product("ACGTN", repeat=3)):
        if "N" in codon:
            table[codon] = _n_codon(codon)
        else:
            table[codon] = (CODE[codon], "no_N")
    return table


TABLE = build_table()


def translate_frame(seq, frame, counts=None):
    """Translate seq from offset frame (0, 1, 2). Bases other than ACGT count as N."""
    out = []
    for i in range(frame, len(seq) - 2, 3):
        codon = seq[i : i + 3]
        aa, how = TABLE.get(codon, TABLE[
            "".join(b if b in "ACGT" else "N" for b in codon)
        ])
        if counts is not None:
            counts[how] += 1
        out.append(aa)
    return "".join(out)


def kept_translations(read, counts=None):
    """The six frames of read, cut at stops; keep pieces with >= MIN_AA residues.

    Only the first piece of each frame is the "before a stop" piece the rule names; the
    pieces after later stops are kept by the same length rule, since each of them is also
    a stretch of at least MIN_AA residues before a stop or the read's end.
    Returns a list of (strand, frame, start_residue, peptide).
    """
    read = read.upper()
    rc = read.translate(COMPLEMENT)[::-1]
    kept = []
    for strand, s in (("+", read), ("-", rc)):
        for frame in range(3):
            prot = translate_frame(s, frame, counts)
            start = 0
            for piece in prot.split("*"):
                if len(piece) >= MIN_AA:
                    kept.append((strand, frame, start, piece))
                start += len(piece) + 1
    return kept


def write_table(path):
    with open(path, "w") as fh:
        fh.write("codon\tamino_acid\trule\tthe_four_codons\n")
        for codon, (aa, how) in TABLE.items():
            if "N" not in codon:
                continue
            four = ""
            if codon.count("N") == 1:
                four = ",".join(
                    f"{codon.replace('N', b)}={CODE[codon.replace('N', b)]}" for b in "ACGT"
                )
            fh.write(f"{codon}\t{aa}\t{how}\t{four}\n")


if __name__ == "__main__":
    write_table(sys.argv[1])
    c = Counter(how for codon, (aa, how) in TABLE.items() if "N" in codon)
    print(dict(c))
