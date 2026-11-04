"""Ask whether a probe's reading frame can be picked by matching k-mers to a reference proteome.

Usage: frame_test.py <probes.fasta[.gz]> <proteome folder> <out.tsv>
The proteome folder holds gencode.v50 and gencode.vM39 pc_translations.fa.gz from GENCODE, and
UP000596742.fa.gz (Mytilus galloprovincialis) and UP000005408.fa.gz (Pacific oyster) from UniProt.

A frame is the six-frame translation. A stretch is the part of a frame between two stop codons.
A frame or stretch passes if it has at least 10 k-mers and at least 90 % of them are in the proteome.
Each sequence is also shuffled three times, with its letters kept, as a chance control.
"""

import collections
import gzip
import random
import sys

import numpy as np

PROBES, DATA, OUT = sys.argv[1:4]
AA = "ACDEFGHIKLMNPQRSTVWY"
HP = {a: (0 if a in "AFGILMPVWY" else 1) for a in AA}
BASES = "TCAG"
CODON = {
    a + b + c: "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"[i * 16 + j * 4 + k]
    for i, a in enumerate(BASES)
    for j, b in enumerate(BASES)
    for k, c in enumerate(BASES)
}


def fasta(p):
    name, buf = None, []
    with gzip.open(p, "rt") if p.endswith(".gz") else open(p) as f:
        for line in f:
            if line[0] == ">":
                if name is not None:
                    yield name, "".join(buf)
                name, buf = line[1:].split()[0], []
            else:
                buf.append(line.strip())
    yield name, "".join(buf)


def encode(seq_list, alpha):
    lut = np.full(256, 255, dtype=np.uint8)
    for i, a in enumerate(AA):
        lut[ord(a)] = i if alpha == "protein" else HP[a]
    return lut[np.frombuffer("#".join(seq_list).encode(), dtype=np.uint8)]


def kmers(codes, k, base):
    n = len(codes) - k + 1
    if n <= 0:
        return np.empty(0, dtype=np.uint64), np.empty(0, dtype=bool)
    bad = np.convolve((codes == 255).astype(np.int64), np.ones(k, dtype=np.int64), "valid") > 0
    val = np.zeros(n, dtype=np.uint64)
    c = np.where(codes == 255, 0, codes).astype(np.uint64)
    for j in range(k):
        val = val * np.uint64(base) + c[j : j + n]
    return val, ~bad


def ref_set(paths, alpha, k):
    base = 20 if alpha == "protein" else 2
    parts = []
    for p in paths:
        seqs = sorted({s for _, s in fasta(f"{DATA}/{p}")})
        v, ok = kmers(encode(seqs, alpha), k, base)
        parts.append(np.unique(v[ok]))
    return np.unique(np.concatenate(parts))


def frames(dna):
    rc = dna[::-1].translate(str.maketrans("ACGT", "TGCA"))
    for s in (dna, rc):
        for f in range(3):
            yield "".join(CODON.get(s[i : i + 3], "X") for i in range(f, len(s) - 2, 3))


def stretch_hits(p, alpha, k, ref):
    base = 20 if alpha == "protein" else 2
    v, ok = kmers(encode([p], alpha), k, base)
    if not ok.any():
        return 0, 0
    v = v[ok]
    i = np.searchsorted(ref, v).clip(max=len(ref) - 1)
    return len(v), int((ref[i] == v).sum())


def score(tags, alpha, k, ref, min_k=10, cut=0.9):
    res = collections.Counter()
    covered = []
    for t in tags:
        whole, stretch, anyhit, best_len = 0, 0, False, 0
        for fr in frames(t):
            parts = [p for p in fr.split("*") if len(p) >= k]
            tot = hit = 0
            for p in parts:
                n, h = stretch_hits(p, alpha, k, ref)
                tot += n
                hit += h
                if h:
                    anyhit = True
                if n >= min_k and h / n >= cut:
                    stretch += 1
                    best_len = max(best_len, len(p))
            if tot >= min_k and hit / tot >= cut:
                whole += 1
        res["whole_frame"] += whole > 0
        res["whole_frame_one"] += whole == 1
        res["stretch"] += stretch > 0
        res["stretch_one"] += stretch == 1
        res["any_kmer"] += anyhit
        if best_len:
            covered.append(3 * best_len / len(t))
    res["cover"] = float(np.median(covered)) if covered else float("nan")
    return res


def mean_over(results, key):
    return round(sum(x[key] for x in results) / len(results), 1)


tags = sorted({s.upper() for _, s in fasta(PROBES)})
print("tags", len(tags), flush=True)
rng = random.Random(20261103)
shuf = [["".join(rng.sample(t, len(t))) for t in tags] for _ in range(3)]
REFS = {
    "same_species_mussel": ["UP000596742.fa.gz"],
    "oyster": ["UP000005408.fa.gz"],
    "human_mouse_gencode": ["gencode.v50.pc_translations.fa.gz", "gencode.vM39.pc_translations.fa.gz"],
}
with open(OUT, "w") as o:
    print(
        "reference\talphabet\tk\tref_distinct_kmers\ttags\ttags_frame_ge90pct\ttags_one_frame_ge90pct\ttags_stretch_ge90pct\ttags_one_stretch_ge90pct\ttags_any_kmer\tmedian_share_of_tag_in_passing_stretch\tshuffled_frame_ge90pct_mean\tshuffled_stretch_ge90pct_mean\tshuffled_any_kmer_mean",
        file=o,
    )
    for name, paths in REFS.items():
        for alpha, k in [("protein", 5), ("protein", 7), ("protein", 10), ("hp", 15), ("hp", 20), ("hp", 30)]:
            ref = ref_set(paths, alpha, k)
            r = score(tags, alpha, k, ref)
            sh = [score(s, alpha, k, ref) for s in shuf]
            row = [
                name,
                alpha,
                k,
                len(ref),
                len(tags),
                r["whole_frame"],
                r["whole_frame_one"],
                r["stretch"],
                r["stretch_one"],
                r["any_kmer"],
                round(r["cover"], 3),
                mean_over(sh, "whole_frame"),
                mean_over(sh, "stretch"),
                mean_over(sh, "any_kmer"),
            ]
            print(*row, sep="\t", file=o, flush=True)
            print(*row, flush=True)
