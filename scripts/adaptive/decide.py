#!/usr/bin/env python3
"""Keep or eject a nanopore read from its first bases, by translated reduced-alphabet k-mers.

Three parts: a translator (six frames, standard genetic code), a set of encoded k-mers built
from reference proteins, and a rule (keep the read if at least --min-kmers distinct chunk
k-mers are in the set). Standard library only. The alphabet tables are copied from
sourmash (src/core/src/encodings.rs, DAYHOFFTABLE and HPTABLE), which kmerseek uses for its
`protein`, `dayhoff` and `hp` encodings (kmerseek src/rust/alphabets.rs, canonical_moltype).

Example:
    decide.py --reference toxins.fasta --reads reads.fastq --chunk 400 \
        --alphabet dayhoff --k 13 --min-kmers 2 > decisions.tsv
"""

import argparse
import gzip
import sys
import time

# sourmash DAYHOFFTABLE: six classes a-f.
DAYHOFF = {
    **dict.fromkeys("C", "a"),
    **dict.fromkeys("AGPST", "b"),
    **dict.fromkeys("DENQ", "c"),
    **dict.fromkeys("HKR", "d"),
    **dict.fromkeys("ILMV", "e"),
    **dict.fromkeys("FWY", "f"),
}
# sourmash HPTABLE (Lehninger): h = hydrophobic, p = polar.
HP = {**dict.fromkeys("AFGILMPVWY", "h"), **dict.fromkeys("NCSTDERHKQ", "p")}
PROTEIN = {aa: aa for aa in "ACDEFGHIKLMNPQRSTVWY"}
ALPHABETS = {"protein": PROTEIN, "dayhoff": DAYHOFF, "hp": HP}

# Standard genetic code, codons in TCAG order.
_BASES = "TCAG"
_AAS = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
CODON = {
    a + b + c: _AAS[16 * i + 4 * j + k]
    for i, a in enumerate(_BASES)
    for j, b in enumerate(_BASES)
    for k, c in enumerate(_BASES)
}
_RC = str.maketrans("ACGTUNacgtun", "TGCAANtgcaan")


def translate(dna):
    """Translate one frame; codons with any base other than ACGT become X."""
    return "".join(CODON.get(dna[i : i + 3], "X") for i in range(0, len(dna) - 2, 3))


def six_frames(dna):
    """The six translations of a DNA string, as (frame label, protein) pairs.

    Frames +1..+3 start at base 0, 1, 2 of the read; -1..-3 at base 0, 1, 2 of its reverse
    complement.
    """
    dna = dna.upper().replace("U", "T")
    rc = dna.translate(_RC)[::-1]
    return [(f"+{f + 1}", translate(dna[f:])) for f in range(3)] + [
        (f"-{f + 1}", translate(rc[f:])) for f in range(3)
    ]


def encode(protein, alphabet):
    """Encode a protein string; residues outside the table (X, *, U, B, Z) become '.'."""
    table = ALPHABETS[alphabet]
    return "".join(table.get(aa, ".") for aa in protein)


def kmers(encoded, k):
    """Every k-mer start position and k-mer that contains no '.', in order."""
    out = []
    bad = -1  # last position of a '.' seen
    for i, c in enumerate(encoded):
        if c == ".":
            bad = i
        if i >= k - 1 and bad < i - k + 1:
            out.append((i - k + 1, encoded[i - k + 1 : i + 1]))
    return out


def build_index(proteins, alphabet, k):
    """The set of encoded k-mers of an iterable of protein strings."""
    index = set()
    for p in proteins:
        index.update(km for _, km in kmers(encode(p, alphabet), k))
    return index


def decide(chunk, index, alphabet, k, min_kmers):
    """Score one chunk against an index.

    Returns (n_hits, best_frame, best_frame_hits, keep). n_hits counts distinct encoded
    k-mers, over all six frames, that are in the index.
    """
    hits = set()
    best_frame, best_n = "", 0
    for frame, prot in six_frames(chunk):
        frame_hits = {km for _, km in kmers(encode(prot, alphabet), k) if km in index}
        if len(frame_hits) > best_n:
            best_frame, best_n = frame, len(frame_hits)
        hits |= frame_hits
    return len(hits), best_frame, best_n, len(hits) >= min_kmers


def read_records(path):
    """Yield (id, sequence) from FASTA or FASTQ, gzipped or not."""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        first = f.read(1)
        if first == "@":
            f.seek(0)
            while True:
                h = f.readline()
                if not h:
                    return
                s = f.readline().strip()
                f.readline()
                f.readline()
                yield h[1:].split()[0], s
        else:
            f.seek(0)
            name, seq = None, []
            for line in f:
                if line.startswith(">"):
                    if name is not None:
                        yield name, "".join(seq)
                    name, seq = line[1:].split()[0], []
                else:
                    seq.append(line.strip())
            if name is not None:
                yield name, "".join(seq)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--reference", required=True, help="protein FASTA to build the k-mer set from"
    )
    p.add_argument(
        "--reads", required=True, help="reads, FASTA or FASTQ, gzipped or not"
    )
    p.add_argument(
        "--chunk", type=int, default=400, help="first N bases of each read to use"
    )
    p.add_argument("--alphabet", choices=sorted(ALPHABETS), default="dayhoff")
    p.add_argument("--k", type=int, required=True, help="k-mer size, in residues")
    p.add_argument(
        "--min-kmers",
        type=int,
        default=1,
        help="keep when at least this many chunk k-mers are in the set",
    )
    a = p.parse_args()

    index = build_index((s for _, s in read_records(a.reference)), a.alphabet, a.k)
    w = sys.stdout
    w.write(
        "read_id\tchunk_bases\tn_kmers_in_index\tbest_frame\tbest_frame_kmers\tkeep\tdecision_seconds\n"
    )
    for rid, seq in read_records(a.reads):
        t0 = time.perf_counter()
        n, frame, n_frame, keep = decide(
            seq[: a.chunk], index, a.alphabet, a.k, a.min_kmers
        )
        dt = time.perf_counter() - t0
        w.write(
            f"{rid}\t{min(len(seq), a.chunk)}\t{n}\t{frame}\t{n_frame}\t{int(keep)}\t{dt:.6f}\n"
        )


if __name__ == "__main__":
    main()
