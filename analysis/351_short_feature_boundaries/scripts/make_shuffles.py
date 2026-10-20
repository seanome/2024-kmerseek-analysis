"""Write the query proteins and 10 shuffles of each, which keep its amino-acid composition.

A shuffle is a random reordering of the protein's own residues, so it has exactly the
same letters in different positions. Shuffle j of protein P is named P__shuf01 .. P__shuf10.

Usage: python -I make_shuffles.py <query_proteins.fasta> <all_queries.fasta> [seed]
"""

import random
import sys

N_SHUFFLES = 10
SEED = 351


def read_fasta(path):
    seqs, name = {}, None
    for line in open(path):
        line = line.strip()
        if line.startswith(">"):
            name = line[1:].split()[0]
            seqs[name] = []
        elif name:
            seqs[name].append(line)
    return {k: "".join(v) for k, v in seqs.items()}


def shuffles(seq, rng, n=N_SHUFFLES):
    out = []
    for _ in range(n):
        letters = list(seq)
        rng.shuffle(letters)
        out.append("".join(letters))
    return out


def main(in_path, out_path, seed=SEED):
    rng = random.Random(seed)
    seqs = read_fasta(in_path)
    with open(out_path, "w") as fh:
        for name in sorted(seqs):
            fh.write(f">{name}\n{seqs[name]}\n")
            for j, s in enumerate(shuffles(seqs[name], rng), start=1):
                fh.write(f">{name}__shuf{j:02d}\n{s}\n")
    print(f"{len(seqs)} proteins, {len(seqs) * N_SHUFFLES} shuffles, seed {seed}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else SEED)
