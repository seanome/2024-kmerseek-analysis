#!/usr/bin/env python3
"""
Dipeptide-preserving shuffle of a protein FASTA (Altschul-Erikson / Euler-path).

Why dipeptide and not monopeptide
---------------------------------
This shuffle is the null for the low-complexity mask, so it has to *keep*
producing the low-complexity k-mers the mask is meant to catch while destroying
all homology. A monopeptide (plain composition-preserving) shuffle scatters a
poly-Q tract's glutamines uniformly across the sequence, dissolving exactly the
local low-complexity runs under study and understating the null hit rate. A
k=2-preserving shuffle holds the count of every dipeptide fixed, so QQ pairs
stay chained and tracts largely survive — which is why dinucleotide/dipeptide
shuffling is the standard null in the low-complexity literature.

Each sequence is shuffled independently, so every domain keeps its own residue
and dipeptide composition (and therefore its own HP composition); only homology
between domains is destroyed. FASTA headers are preserved verbatim so that
SCOP labels can still be joined onto shuffled hits — a shuffled run should show
no fold-level enrichment, which is the null's validity check.

Algorithm
---------
Kandel et al. (1996) / Altschul & Erikson (1985): the sequence is a Eulerian
path through a multigraph whose vertices are residues and whose edges are the
adjacent pairs. Uniformly sampling Eulerian paths with the same first and last
vertex yields a uniform sample over sequences with identical dipeptide counts.
Sampling is done by choosing, for each vertex, a random "last edge" such that
the chosen edges form an arborescence rooted at the final vertex (rejection
sampling until they do), then randomly permuting each vertex's remaining edges.

Usage:
    shuffle_fasta.py --input in.fa --output out.fa --seed 1 [--shuffle-k 2]
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter, defaultdict


def parse_fasta(path):
    """Yield (header, sequence) pairs. Header excludes the leading '>'."""
    header, chunks = None, []
    with open(path) as handle:
        for line in handle:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(chunks)
                header, chunks = line[1:], []
            else:
                chunks.append(line.strip())
    if header is not None:
        yield header, "".join(chunks)


def _forms_arborescence(last_edge: dict[str, str], root: str) -> bool:
    """True if following `last_edge` from every vertex terminates at `root`."""
    for start in last_edge:
        seen = set()
        node = start
        while node != root:
            if node in seen or node not in last_edge:
                return False  # cycle, or dead end that never reaches root
            seen.add(node)
            node = last_edge[node]
    return True


def shuffle_dipeptide(seq: str, rng: random.Random, max_tries: int = 1000) -> str:
    """Shuffle preserving exact dipeptide (2-mer) counts, plus first/last residue."""
    if len(seq) < 3:
        return seq

    root = seq[-1]
    out_edges: dict[str, list[str]] = defaultdict(list)
    for a, b in zip(seq, seq[1:]):
        out_edges[a].append(b)

    # Rejection-sample a last-edge choice per vertex until they form a tree
    # rooted at the final residue, which is exactly the condition for the
    # resulting edge ordering to admit a complete Eulerian path.
    for _ in range(max_tries):
        last_edge = {
            v: rng.choice(targets)
            for v, targets in out_edges.items()
            if v != root and targets
        }
        if _forms_arborescence(last_edge, root):
            break
    else:
        # Astronomically unlikely for real protein sequences; fall back to the
        # identity rather than emit a sequence with the wrong composition.
        print(
            f"warning: Euler sampling failed after {max_tries} tries "
            f"(len={len(seq)}); leaving sequence unshuffled",
            file=sys.stderr,
        )
        return seq

    # Randomly order each vertex's edges, with its chosen last edge pinned last.
    ordered: dict[str, list[str]] = {}
    for v, targets in out_edges.items():
        remaining = list(targets)
        if v in last_edge:
            remaining.remove(last_edge[v])
            rng.shuffle(remaining)
            remaining.append(last_edge[v])
        else:
            rng.shuffle(remaining)
        ordered[v] = remaining

    # Walk the Eulerian path, consuming each vertex's edges in the order above.
    cursor: dict[str, int] = defaultdict(int)
    node = seq[0]
    result = [node]
    for _ in range(len(seq) - 1):
        idx = cursor[node]
        cursor[node] = idx + 1
        node = ordered[node][idx]
        result.append(node)
    return "".join(result)


def shuffle_monopeptide(seq: str, rng: random.Random) -> str:
    """Plain composition-preserving shuffle (--shuffle-k 1), for contrast."""
    letters = list(seq)
    rng.shuffle(letters)
    return "".join(letters)


def dipeptide_counts(seq: str) -> Counter:
    return Counter(seq[i : i + 2] for i in range(len(seq) - 1))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--shuffle-k",
        type=int,
        default=2,
        choices=(1, 2),
        help="2 = preserve dipeptide counts (default); 1 = preserve composition only",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="write a verification report here (composition checks per sequence)",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    n_seqs = 0
    n_dipeptide_ok = 0
    n_composition_ok = 0
    n_unchanged = 0
    n_short = 0

    with open(args.output, "w") as out:
        for header, seq in parse_fasta(args.input):
            n_seqs += 1
            if args.shuffle_k == 2:
                shuffled = shuffle_dipeptide(seq, rng)
                if dipeptide_counts(shuffled) == dipeptide_counts(seq):
                    n_dipeptide_ok += 1
            else:
                shuffled = shuffle_monopeptide(seq, rng)

            if Counter(shuffled) == Counter(seq):
                n_composition_ok += 1
            if len(seq) < 3:
                n_short += 1
            elif shuffled == seq:
                n_unchanged += 1

            out.write(f">{header}\n")
            for i in range(0, len(shuffled), 60):
                out.write(shuffled[i : i + 60] + "\n")

    # These are invariants, not diagnostics: a shuffle that changed composition
    # would invalidate every downstream complexity comparison, so fail loudly.
    if n_composition_ok != n_seqs:
        print(
            f"FATAL: residue composition changed in {n_seqs - n_composition_ok}/{n_seqs} sequences",
            file=sys.stderr,
        )
        return 1
    if args.shuffle_k == 2 and n_dipeptide_ok != n_seqs:
        print(
            f"FATAL: dipeptide composition changed in {n_seqs - n_dipeptide_ok}/{n_seqs} sequences",
            file=sys.stderr,
        )
        return 1

    lines = [
        f"input\t{args.input}",
        f"seed\t{args.seed}",
        f"shuffle_k\t{args.shuffle_k}",
        f"n_sequences\t{n_seqs}",
        f"n_composition_preserved\t{n_composition_ok}",
        f"n_dipeptide_preserved\t{n_dipeptide_ok if args.shuffle_k == 2 else 'NA'}",
        f"n_identical_to_input\t{n_unchanged}",
        f"n_too_short_to_shuffle\t{n_short}",
    ]
    report = "\n".join(lines) + "\n"
    if args.report:
        with open(args.report, "w") as handle:
            handle.write(report)
    print(report, file=sys.stderr, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
