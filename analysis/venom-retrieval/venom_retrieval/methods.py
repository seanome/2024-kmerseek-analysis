"""The retrieval methods, and the ablations that take them apart.

Five methods are compared, each scoring every reference peptide for one query and
returning a ranking:

  framework_only          the cysteine framework as a label: same framework or not.
  loop_lengths_only       the inter-cysteine loop lengths as numbers, graded by distance.
  loop_composition_only   which residues the loops hold, with their order thrown away.
  loop_anchored           the loops aligned residue by residue, loop i against loop i.
  full_sequence           Smith-Waterman over the whole peptide, affine gaps, BLOSUM62.
  kmer_<alphabet>_k<k>    reduced-alphabet k-mer containment, every k-mer kept.
  framework_then_rerank   framework-compatible candidates only, reranked by full_sequence.

The first five are also the ablation ladder, in that order: each one adds one kind of
information to the one before it. Framework only knows the pattern; loop lengths know the
numbers; loop composition knows which residues are present; loop-anchored alignment knows
their order inside each loop; full sequence knows the order of everything including the
cysteines. If the ladder is flat, the framework is doing the work. If only the top rung
performs, the framework is not.

Two rules hold throughout, because breaking either would make a result mean less than it
appears to:

  Nothing infers disulfide connectivity from cysteine order. `loop_anchored` lines up the
  cysteines in sequence order, which is an ordering, not a bonding pattern, and two
  peptides with identical frameworks can be bonded differently.

  No method forces peptides with different cysteine counts into an alignment. Where the
  counts differ, the loop-wise methods return no score at all rather than a bad one, and
  the evaluation counts a true positive lost that way as an end-to-end miss, not as
  something outside the method's remit.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from . import encodings, frameworks
from .align import local_score

#: The residues a composition vector counts, in a fixed order.
COMPOSITION_ALPHABET = encodings.CANONICAL_AA

#: Terminal segments are part of the loop-wise comparison, as two further segments: the
#: stretch before the first cysteine and the stretch after the last. They are included
#: rather than dropped because in several venom families the C-terminal tail is where the
#: activity-determining residues sit. `LoopFeatures.segments` holds them at the two ends,
#: and `include_termini=False` is what documents their exclusion if that is ever wanted.
INCLUDE_TERMINI = True


@dataclass
class PeptideFeatures:
    """Everything the methods need about one peptide, computed once."""

    peptide_id: str
    sequence: str
    framework: str
    n_cysteines: int
    loop_lengths: tuple[int, ...]
    #: The inter-cysteine loops, with the N- and C-terminal stretches at the two ends when
    #: `INCLUDE_TERMINI`. The cysteines themselves are not in any segment.
    segments: tuple[str, ...]
    composition: np.ndarray
    self_score: float
    loop_self_score: float
    kmers: dict[tuple[str, int], frozenset[str]] = field(default_factory=dict)


def composition_vector(seq: str) -> np.ndarray:
    """The share of each of the 20 residues in `seq`, summing to 1 (all zero if empty)."""
    counts = Counter(seq)
    vector = np.array(
        [counts.get(residue, 0) for residue in COMPOSITION_ALPHABET], dtype=float
    )
    total = vector.sum()
    return vector / total if total else vector


def build_features(
    peptide_ids: list[str],
    sequences: list[str],
    kmer_settings: list[tuple[str, int]],
    include_termini: bool = INCLUDE_TERMINI,
) -> list[PeptideFeatures]:
    """Precompute every method's view of every peptide.

    `kmer_settings` is the fixed list of (alphabet, k) pairs, declared in the config.
    Every k-mer of every peptide is kept: no sketch, no hash threshold, which is what
    `scaled=1` means in kmerseek. For a 25-residue peptide a sketch would keep one or two
    k-mers, and the comparison would be mostly chance.
    """
    out = []
    for peptide_id, sequence in zip(peptide_ids, sequences):
        layout = frameworks.loop_layout(sequence)
        segments = layout.loops
        if include_termini:
            segments = (layout.nterm,) + segments + (layout.cterm,)
        out.append(
            PeptideFeatures(
                peptide_id=peptide_id,
                sequence=sequence,
                framework=frameworks.framework_string(sequence),
                n_cysteines=layout.n_cysteines,
                loop_lengths=layout.loop_lengths,
                segments=segments,
                composition=composition_vector("".join(layout.loops)),
                self_score=local_score(sequence, sequence),
                loop_self_score=sum(local_score(s, s) for s in segments),
                kmers={
                    (alphabet, k): frozenset(
                        encodings.kmer_set(sequence, alphabet, k)
                    )
                    for alphabet, k in kmer_settings
                },
            )
        )
    return out


#: Returned instead of a score when a method cannot compare the pair at all, as distinct
#: from comparing it and finding nothing in common. A peptide scored NOT_COMPARABLE is
#: never ranked, and a true positive among them is an end-to-end miss.
NOT_COMPARABLE = float("-inf")


def framework_only(query: PeptideFeatures, target: PeptideFeatures) -> float:
    """1.0 for the same cysteine framework, 0.0 for any other framework.

    The framework as venom-peptide nomenclature uses it: a label, with no notion of two
    frameworks being nearly the same. Every peptide sharing the query's framework ties, so
    this method produces one large tied block rather than a ranking, which is exactly what
    a reviewer working from framework alone would face.
    """
    return 1.0 if query.framework == target.framework and query.framework else 0.0


def loop_lengths_only(query: PeptideFeatures, target: PeptideFeatures) -> float:
    """Graded by how far the loop lengths are apart, ignoring every residue.

    `1 / (1 + total difference)`, so identical loop lengths score 1.0 and each residue of
    difference costs. Not comparable when the cysteine counts differ, because then the
    loops do not correspond.
    """
    if query.n_cysteines != target.n_cysteines or not query.n_cysteines:
        return NOT_COMPARABLE
    distance = sum(
        abs(a - b) for a, b in zip(query.loop_lengths, target.loop_lengths)
    )
    return 1.0 / (1.0 + distance)


def loop_composition_only(query: PeptideFeatures, target: PeptideFeatures) -> float:
    """Cosine similarity of the residue composition of the loops, order discarded.

    The loops' residues pooled into one composition vector, so two peptides holding the
    same residues in a different order score the same as if they matched. This is the rung
    of the ladder that says whether residue order matters at all.
    """
    if query.n_cysteines != target.n_cysteines or not query.n_cysteines:
        return NOT_COMPARABLE
    a, b = query.composition, target.composition
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b) / norm if norm else 0.0


def loop_anchored(query: PeptideFeatures, target: PeptideFeatures) -> float:
    """Loop i of the query aligned against loop i of the target, scores summed.

    The cysteines are the anchors: they are taken as corresponding in order, and only the
    stretches between them are aligned, each against its counterpart, with the same affine
    gaps and BLOSUM62 as the full-sequence method. Insertions and deletions inside a loop
    are therefore scored, and a loop that has grown or shrunk is penalised by the gap
    costs rather than by an arbitrary rule.

    The terminal stretches are two further segments, included (see `INCLUDE_TERMINI`).

    Not comparable when the cysteine counts differ. Nothing is stretched or padded to make
    incompatible frameworks line up, and nothing here is a statement about which cysteines
    are bonded to which.

    The sum is divided by the query's own loop self-score, so the result is on the same
    0-to-1 scale as the full-sequence method and peptides of different lengths compare.
    """
    if query.n_cysteines != target.n_cysteines or not query.n_cysteines:
        return NOT_COMPARABLE
    if query.loop_self_score <= 0:
        return 0.0
    total = sum(
        local_score(a, b) for a, b in zip(query.segments, target.segments)
    )
    return total / query.loop_self_score


def full_sequence(query: PeptideFeatures, target: PeptideFeatures) -> float:
    """Smith-Waterman over the whole mature peptide, divided by the query's self-score."""
    if query.self_score <= 0:
        return 0.0
    return local_score(query.sequence, target.sequence) / query.self_score


def kmer_containment(alphabet: str, ksize: int):
    """A method that scores by the share of the query's k-mers the target holds."""

    def score(query: PeptideFeatures, target: PeptideFeatures) -> float:
        mine = query.kmers[(alphabet, ksize)]
        if not mine:
            # The peptide is shorter than k, so this setting has nothing to say about it.
            return NOT_COMPARABLE
        theirs = target.kmers[(alphabet, ksize)]
        return len(mine & theirs) / len(mine)

    score.__name__ = f"kmer_{alphabet}_k{ksize}"
    score.__doc__ = (
        f"The share of the query's {alphabet} k={ksize} k-mers the target also holds. "
        f"Every k-mer is kept; there is no sketch and no hash threshold."
    )
    return score


def framework_candidates(
    query: PeptideFeatures,
    targets: list[PeptideFeatures],
    max_loop_difference: int | None,
) -> list[int]:
    """The indices of targets whose framework is compatible with the query's.

    Compatible means the same number of cysteines, so the anchors correspond in order,
    and, where `max_loop_difference` is set, every pair of corresponding loops within that
    many residues. This is the candidate-generation step of `framework_then_rerank`, and
    the evaluation measures separately how many true positives it throws away.
    """
    out = []
    for index, target in enumerate(targets):
        if query.n_cysteines != target.n_cysteines or not query.n_cysteines:
            continue
        if max_loop_difference is not None and any(
            abs(a - b) > max_loop_difference
            for a, b in zip(query.loop_lengths, target.loop_lengths)
        ):
            continue
        out.append(index)
    return out


def rank(
    query: PeptideFeatures,
    targets: list[PeptideFeatures],
    score,
    candidates: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Score the targets and sort them best first.

    Returns `(indices, scores)` over the targets this method could compare. Targets it
    could not compare, and targets outside `candidates` where a candidate list is given,
    are left out of both arrays entirely rather than being given a low score: a method
    that cannot judge a peptide has not judged it badly, and keeping the two apart is what
    lets the evaluation report candidate-generation loss separately.

    Ties keep the order the targets came in, which is identifier order, so a run is
    reproducible even where a method produces one large tied block.
    """
    pool = range(len(targets)) if candidates is None else candidates
    kept, values = [], []
    for index in pool:
        value = score(query, targets[index])
        if value == NOT_COMPARABLE:
            continue
        kept.append(index)
        values.append(value)
    if not kept:
        return np.empty(0, dtype=int), np.empty(0, dtype=float)
    indices = np.array(kept)
    scores = np.array(values, dtype=float)
    order = np.argsort(-scores, kind="stable")
    return indices[order], scores[order]
