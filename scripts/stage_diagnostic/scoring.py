"""Seedless local alignment with affine gaps: the full-residue scorer (stage A) and the
reduced-alphabet scorer (stage B, also used to rescore kmerseek's candidates in C and D).

Both are Smith-Waterman via Biopython's PairwiseAligner, every query against every
reference it is asked about, no seeds and no filters.

Full scorer: BLOSUM62 (half-bit units), gap open 11, extend 1, as BLAST charges them
(a gap of n residues costs 11 + n).

Reduced scorer: BLOSUM62 collapsed onto the alphabet's classes. BLOSUM62's scores are
s_ij = 2 log2(p_ij / (q_i q_j)), so the pair frequencies are p_ij = q_i q_j 2^(s_ij / 2)
(renormalised to sum to 1). Summing them inside classes gives P_AB and Q_A, and the class
score is S_AB = 2 log2(P_AB / (Q_A Q_B)), in the same half-bit units, with the same gap
costs. Nothing in it is fitted to the benchmark. If the collapsed matrix's expected score
sum_AB Q_A Q_B S_AB is not negative, local alignment has no length-independent score and
the matrix is flagged (it is still run, and the flag is reported).
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
from Bio.Align import PairwiseAligner, substitution_matrices

from alphabets import CANONICAL, CLUSTERS, class_letters, encode

# Background residue frequencies used to undo BLOSUM62's log-odds. These are the row sums
# of the BLOSUM62 target frequencies (Henikoff & Henikoff 1992) as distributed with NCBI
# BLAST; an error of a few percent here moves the collapsed scores by well under a half-bit.
BLOSUM62_BACKGROUND = dict(
    A=0.0740, R=0.0516, N=0.0446, D=0.0536, C=0.0247, Q=0.0342, E=0.0543, G=0.0741,
    H=0.0262, I=0.0679, L=0.0989, K=0.0580, M=0.0250, F=0.0474, P=0.0388, S=0.0573,
    T=0.0507, W=0.0134, Y=0.0321, V=0.0729,
)  # fmt: skip

GAP_OPEN = 11
GAP_EXTEND = 1


@lru_cache(maxsize=None)
def blosum62():
    return substitution_matrices.load("BLOSUM62")


@lru_cache(maxsize=None)
def collapsed_matrix(alphabet: str):
    """(Biopython Array over class letters, expected score per aligned pair)."""
    b = blosum62()
    q = np.array([BLOSUM62_BACKGROUND[r] for r in CANONICAL])
    q = q / q.sum()
    s = np.array([[b[a][c] for c in CANONICAL] for a in CANONICAL], dtype=float)
    p = np.outer(q, q) * np.power(2.0, s / 2.0)
    p = p / p.sum()
    letters = class_letters(alphabet)
    idx = {r: i for i, r in enumerate(CANONICAL)}
    members = [[idx[r] for r in cl] for cl in CLUSTERS[alphabet]]
    n = len(letters)
    S = np.zeros((n, n))
    Q = np.array([q[m].sum() for m in members])
    for i in range(n):
        for j in range(n):
            P = p[np.ix_(members[i], members[j])].sum()
            S[i, j] = 2 * math.log2(P / (Q[i] * Q[j]))
    expected = float((np.outer(Q, Q) * S).sum())
    arr = substitution_matrices.Array(alphabet=letters, dims=2, data=S)
    return arr, expected


def _aligner(matrix) -> PairwiseAligner:
    a = PairwiseAligner()
    a.mode = "local"
    a.substitution_matrix = matrix
    a.open_gap_score = -(GAP_OPEN + GAP_EXTEND)
    a.extend_gap_score = -GAP_EXTEND
    return a


class Scorer:
    """`full` (stage A) or a reduced alphabet name (stage B)."""

    def __init__(self, kind: str):
        self.kind = kind
        if kind == "full":
            self.aligner = _aligner(blosum62())
            self.expected = None
        else:
            m, self.expected = collapsed_matrix(kind)
            self.aligner = _aligner(m)

    def prep(self, seq: str) -> str:
        return seq if self.kind == "full" else encode(seq, self.kind)

    def score(self, q: str, t: str) -> float:
        """q and t already passed through prep()."""
        return float(self.aligner.score(q, t))

    def coords(self, q: str, t: str) -> tuple[int, int, int, int]:
        """0-based, end-exclusive (query_start, query_end, target_start, target_end) of the
        best local alignment. Ties go to Biopython's first alignment."""
        aln = self.aligner.align(q, t)[0]
        tb, qb = aln.aligned[1], aln.aligned[0]
        return int(qb[0][0]), int(qb[-1][1]), int(tb[0][0]), int(tb[-1][1])
