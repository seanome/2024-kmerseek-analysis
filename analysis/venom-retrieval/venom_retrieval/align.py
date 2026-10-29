"""Full-amino-acid local alignment with affine gaps, and the loop-wise variant.

One aligner, built once per process, used by three things: the full-sequence comparison
(method B), the reranking step (method D), and the per-loop comparison (method E). They
share it so that a difference between them is a difference in what is aligned, not in how.

The settings are BLAST's protein defaults: BLOSUM62, gap open -11, gap extend -1, local
(Smith-Waterman) mode. Nothing here is tuned; they are written down once and used
everywhere.

Biopython's `PairwiseAligner.score` computes the score without a traceback, which is all
any method here needs, and is about fifteen times faster than producing the alignment.
"""

from __future__ import annotations

from functools import lru_cache

from Bio import Align
from Bio.Align import substitution_matrices

#: BLAST's protein defaults. Written as a tuple so a caller can print what ran.
MATRIX = "BLOSUM62"
GAP_OPEN = -11.0
GAP_EXTEND = -1.0


@lru_cache(maxsize=1)
def aligner() -> Align.PairwiseAligner:
    """The one local aligner, built on first use in each process."""
    return Align.PairwiseAligner(
        mode="local",
        substitution_matrix=substitution_matrices.load(MATRIX),
        open_gap_score=GAP_OPEN,
        extend_gap_score=GAP_EXTEND,
    )


def local_score(query: str, target: str) -> float:
    """The Smith-Waterman score of `query` against `target`, 0 for an empty sequence.

    >>> local_score("ACDEF", "ACDEF") > local_score("ACDEF", "KKKKK")
    True
    >>> local_score("", "ACDEF")
    0.0
    """
    if not query or not target:
        return 0.0
    return float(aligner().score(query, target))


def self_scores(sequences: list[str]) -> list[float]:
    """Each sequence's score against itself, the natural scale for its own comparisons."""
    return [local_score(seq, seq) for seq in sequences]


def normalised_local_score(query: str, target: str) -> float:
    """`local_score` divided by the smaller of the two self-scores.

    1.0 for identical sequences, and comparable between pairs of different lengths, which
    a raw score is not: a long peptide scores higher against anything simply by being
    long.

    >>> round(normalised_local_score("ACDEFGH", "ACDEFGH"), 3)
    1.0
    """
    smaller = min(local_score(query, query), local_score(target, target))
    if smaller <= 0:
        return 0.0
    return local_score(query, target) / smaller
