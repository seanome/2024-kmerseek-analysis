"""Protein sequences that keep every reduced k-mer but change the order (notebook 343).

A reduced string is a protein written in a reduced alphabet (one letter per residue class,
for example h/p for hydrophobic/polar). Its length-k words are the only thing kmerseek
hashes. This module builds other reduced strings with the same length-k words, each word
the same number of times, and then writes an amino-acid sequence for each one from the
protein's own residues.

The construction is the de Bruijn graph of the reduced string: one node per length-(k-1)
word, one edge per length-k word occurrence, from its first k-1 letters to its last k-1
letters. The original string is a walk that uses every edge once (an Eulerian path). Every
other such walk from the same first node spells a string with the same length-k word
counts. The BEST theorem (de Bruijn, van Aardenne-Ehrenfest, Smith, Tutte) counts these
walks, and the same bijection lets us draw one uniformly at random.
"""

from __future__ import annotations

import random
from collections import Counter
from math import lgamma, log

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from hp_conservation_utils import ALPHABET_CLUSTERS

#: Letters kmerseek uses for the two H/P classes (kmerseek src/rust/alphabets.rs).
HP_LETTERS = "hp"

#: sourmash's own Dayhoff letters, keyed by the residues in each class
#: (sourmash src/core/src/encodings.rs, aa_to_dayhoff). kmerseek hashes Dayhoff through
#: sourmash, so these are the letters its sketches are built from.
DAYHOFF_LETTERS = {
    "C": "a",
    "AGPST": "b",
    "DENQ": "c",
    "HKR": "d",
    "ILMV": "e",
    "FWY": "f",
}

STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")


def encoding_table(alphabet: str) -> dict[str, str]:
    """Map each of the 20 amino acids to its one-letter class code for `alphabet`."""
    clusters = ALPHABET_CLUSTERS[alphabet]
    if alphabet == "dayhoff6":
        letters = [DAYHOFF_LETTERS[c] for c in clusters]
    elif len(clusters) == 2:
        letters = list(HP_LETTERS)
    else:
        letters = [chr(ord("a") + i) for i in range(len(clusters))]
    return {aa: letter for cluster, letter in zip(clusters, letters) for aa in cluster}


def encode(protein: str, alphabet: str) -> str:
    """Write `protein` in the reduced alphabet, one letter per residue."""
    table = encoding_table(alphabet)
    return "".join(table[aa] for aa in protein)


def kmer_counts(s: str, k: int) -> Counter:
    """Count every length-k word of `s`, overlapping."""
    return Counter(s[i : i + k] for i in range(len(s) - k + 1))


class WordGraph:
    """The de Bruijn graph of one reduced string at word length k.

    `succ[v]` lists the successor node of every edge leaving node v, one entry per word
    occurrence, so repeated words appear as repeated entries. Node ids are integers;
    `label[v]` is the last letter of node v's (k-1)-letter word, which is the letter an
    edge into v adds to the string.
    """

    def __init__(self, reduced: str, k: int):
        if len(reduced) < k:
            raise ValueError(
                f"string of length {len(reduced)} has no words of length {k}"
            )
        self.reduced = reduced
        self.k = k
        ids: dict[str, int] = {}
        words = [reduced[i : i + k - 1] for i in range(len(reduced) - k + 2)]
        node = [ids.setdefault(w, len(ids)) for w in words]
        self.n_nodes = len(ids)
        self.label = [""] * self.n_nodes
        for w, i in ids.items():
            self.label[i] = w[-1]
        self.succ: list[list[int]] = [[] for _ in range(self.n_nodes)]
        for a, b in zip(node[:-1], node[1:]):
            self.succ[a].append(b)
        self.start = node[0]
        self.end = node[-1]
        self.prefix = words[0]

    def log_n_strings(self) -> float:
        """Natural log of the number of distinct strings with these word counts.

        Only strings that start with the original first k-1 letters are counted (they then
        also end with the original last k-1 letters). BEST theorem for a walk from s to t:
        walks with distinguishable edges = T_t * prod_{v != t} (out(v) - 1)! * out(t)!,
        where T_t is the number of spanning trees directed towards t (matrix-tree theorem:
        determinant of the out-degree Laplacian with row and column t removed). Parallel
        edges are the same word, so each string is counted prod_e m_e! times.
        """
        t = self.end
        n = self.n_nodes
        out = [len(s) for s in self.succ]
        log_walks = sum(lgamma(out[v]) for v in range(n) if v != t and out[v] > 0)
        log_walks += lgamma(out[t] + 1)
        mult = Counter((a, b) for a in range(n) for b in self.succ[a])
        log_walks -= sum(lgamma(m + 1) for m in mult.values())
        if n == 1:
            return log_walks
        rows, cols, vals = [], [], []
        for (a, b), m in mult.items():
            if a != b:
                rows.append(a)
                cols.append(b)
                vals.append(-float(m))
        for a in range(n):
            loops = mult.get((a, a), 0)
            rows.append(a)
            cols.append(a)
            vals.append(float(out[a] - loops))
        lap = sp.csc_matrix((vals, (rows, cols)), shape=(n, n))
        keep = np.array([v for v in range(n) if v != t])
        minor = lap[keep][:, keep].tocsc()
        lu = spla.splu(minor, permc_spec="COLAMD")
        diag = lu.U.diagonal()
        log_tree = float(np.sum(np.log(np.abs(diag))))
        return log_walks + log_tree

    def sample(self, rng: random.Random) -> str:
        """One string drawn uniformly from all strings with these word counts.

        Draw a uniform spanning tree directed towards the end node with Wilson's
        loop-erased random walk, use its edge as each node's last exit, put the other exits
        in random order, and follow them from the start node (the BEST bijection).
        """
        succ, t, n = self.succ, self.end, self.n_nodes
        in_tree = [False] * n
        in_tree[t] = True
        nxt = [-1] * n
        for v in range(n):
            u = v
            while not in_tree[u]:
                i = rng.randrange(len(succ[u]))
                nxt[u] = i
                u = succ[u][i]
            u = v
            while not in_tree[u]:
                in_tree[u] = True
                u = succ[u][nxt[u]]
        order: list[list[int]] = []
        for v in range(n):
            idx = list(range(len(succ[v])))
            if v != t:
                last = idx.pop(nxt[v])
                rng.shuffle(idx)
                idx.append(last)
            else:
                rng.shuffle(idx)
            idx.reverse()  # pop() from the end takes them in drawn order
            order.append(idx)
        out = [self.prefix]
        v = self.start
        label = self.label
        while order[v]:
            v = succ[v][order[v].pop()]
            out.append(label[v])
        s = "".join(out)
        if len(s) != len(self.reduced):
            raise RuntimeError("walk stopped before using every word")
        return s


def fraction_changed(a: str, b: str) -> float:
    """Fraction of positions where two equal-length strings differ."""
    x = np.frombuffer(a.encode(), dtype=np.uint8)
    y = np.frombuffer(b.encode(), dtype=np.uint8)
    return float(np.mean(x != y))


def realize_protein(protein: str, reduced: str, alt_reduced: str) -> str:
    """Write an amino-acid sequence that encodes to `alt_reduced`, from `protein`'s residues.

    A position whose class did not change keeps its own residue. The residues freed at
    positions whose class changed are placed, in their original N-to-C order, into the
    positions that newly need that class. Both strings have the same count of every class
    letter, so each class has exactly as many freed residues as open positions: the
    amino-acid composition is kept exactly, and so are the first and last k-1 residues,
    which the walk never changes.
    """
    out = list(protein)
    freed: dict[str, list[str]] = {}
    for aa, r, r2 in zip(protein, reduced, alt_reduced):
        if r != r2:
            freed.setdefault(r, []).append(aa)
    pos = {c: 0 for c in freed}
    for i, (r, r2) in enumerate(zip(reduced, alt_reduced)):
        if r != r2:
            out[i] = freed[r2][pos[r2]]
            pos[r2] += 1
    return "".join(out)


def block_shuffle(
    reduced: str, protein: str, block: int, rng: random.Random
) -> tuple[str, str]:
    """Cut both strings into blocks of `block` letters and reorder the inner blocks.

    The first and last blocks stay in place, so both ends are kept, and residues move with
    their reduced letters, so the amino-acid composition is kept. Returns the reduced and
    the amino-acid strings.
    """
    n = len(reduced)
    cuts = list(range(0, n, block)) + [n]
    blocks = list(zip(cuts[:-1], cuts[1:]))
    if len(blocks) < 4:
        return reduced, protein
    inner = blocks[1:-1]
    rng.shuffle(inner)
    order = [blocks[0]] + inner + [blocks[-1]]
    return (
        "".join(reduced[a:b] for a, b in order),
        "".join(protein[a:b] for a, b in order),
    )


def kmer_multiset_change(a: str, b: str, k: int) -> float:
    """Fraction of `a`'s length-k word occurrences not matched by a word occurrence in `b`."""
    ca, cb = kmer_counts(a, k), kmer_counts(b, k)
    shared = sum((ca & cb).values())
    return 1.0 - shared / max(1, sum(ca.values()))


def evaluate_protein(
    protein: str,
    alphabet: str,
    k: int,
    seed: int,
    *,
    n_draws: int,
    n_needed: int,
    min_changed: float,
) -> dict:
    """Test one protein at one alphabet and word length, for both constructions.

    Draws up to `n_draws` strings from each construction, stopping early once `n_needed`
    distinct strings pass. A string passes when it differs from the original at a fraction
    of at least `min_changed` reduced positions, has the original's exact length-k word
    counts, and its amino-acid sequence has the original's exact amino-acid counts.
    """
    reduced = encode(protein, alphabet)
    g = WordGraph(reduced, k)
    log_n = g.log_n_strings()
    rng = random.Random(seed)
    words = kmer_counts(reduced, k)
    aa_counts = Counter(protein)

    def passes(alt_red: str, alt_aa: str) -> tuple[bool, bool]:
        exact = kmer_counts(alt_red, k) == words and Counter(alt_aa) == aa_counts
        if exact and encode(alt_aa, alphabet) != alt_red:
            raise RuntimeError(
                "amino-acid sequence does not encode to the reduced string"
            )
        return exact, exact and fraction_changed(reduced, alt_red) >= min_changed

    row = {"alphabet": alphabet, "k": k, "length": len(protein), "log_n_strings": log_n}
    for method in ("word_walk", "block_shuffle"):
        seen: set[str] = set()
        passing: set[str] = set()
        changed: list[float] = []
        n_exact = 0
        word_change: list[float] = []
        draws = 0
        termini_kept = True
        for draws in range(1, n_draws + 1):
            if method == "word_walk":
                alt_red = g.sample(rng)
                alt_aa = realize_protein(protein, reduced, alt_red)
            else:
                alt_red, alt_aa = block_shuffle(reduced, protein, k, rng)
                word_change.append(kmer_multiset_change(reduced, alt_red, k))
            changed.append(fraction_changed(reduced, alt_red))
            exact, ok = passes(alt_red, alt_aa)
            n_exact += exact and alt_red != reduced
            if method == "word_walk":
                if not exact:
                    raise RuntimeError("word walk broke the word counts")
                termini_kept &= (
                    alt_aa[: k - 1] == protein[: k - 1]
                    and alt_aa[-(k - 1) :] == protein[-(k - 1) :]
                )
            if alt_red != reduced:
                seen.add(alt_red)
            if ok:
                passing.add(alt_red)
            if len(passing) >= n_needed:
                break
        row[f"{method}_draws"] = draws
        row[f"{method}_distinct"] = len(seen)
        row[f"{method}_passing"] = len(passing)
        row[f"{method}_exact_changed_draws"] = n_exact
        row[f"{method}_median_changed"] = float(np.median(changed))
        if method == "word_walk":
            row["word_walk_termini_kept"] = termini_kept
        else:
            row["block_shuffle_median_word_change"] = float(np.median(word_change))
    return row


def n_strings_at_least(log_n: float, n: int) -> bool:
    """True when e**log_n, the number of strings including the original, is at least n."""
    return log_n >= log(n) - 1e-6
