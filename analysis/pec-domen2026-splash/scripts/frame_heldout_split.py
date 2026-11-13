#!/usr/bin/env python3
"""Split the new-genome genes in half at random, for a k-mer reference that lacks the read's own gene.

Usage: frame_heldout_split.py <protein.fasta> <out_dir>

Writes <out_dir>/halfA.faa (every protein of the genes in half A) and <out_dir>/genes_halfB.txt.
A gene's isoforms stay in the same half: the gene is the first word of the header with any "#n"
(isoform number added by sra_frame_test_prepare.py) and any funannotate transcript suffix "-T<n>"
removed. Seed 361.
"""
import random
import re
import sys
from pathlib import Path

SEED = 361


def gene_of(name):
    return re.sub(r"-T\d+$", "", name.split("#")[0])


def main():
    prot_p, out = sys.argv[1], Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    recs, name, buf = [], None, []
    for line in open(prot_p):
        if line.startswith(">"):
            if name:
                recs.append((name, buf))
            name, buf = line[1:].split()[0], [line]
        else:
            buf.append(line)
    if name:
        recs.append((name, buf))
    genes = sorted({gene_of(n) for n, _ in recs})
    rng = random.Random(SEED)
    rng.shuffle(genes)
    half_a = set(genes[: len(genes) // 2])
    n_a = 0
    with open(out / "halfA.faa", "w") as f:
        for n, b in recs:
            if gene_of(n) in half_a:
                f.writelines(b)
                n_a += 1
    (out / "genes_halfB.txt").write_text("".join(g + "\n" for g in sorted(set(genes) - half_a)))
    print(f"proteins {len(recs)}, genes {len(genes)}; half A: {len(half_a)} genes, {n_a} proteins; "
          f"half B: {len(genes) - len(half_a)} genes, {len(recs) - n_a} proteins")


if __name__ == "__main__":
    main()
