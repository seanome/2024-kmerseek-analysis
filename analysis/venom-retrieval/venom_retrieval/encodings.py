"""Reduced amino-acid alphabets, k-mer sets, and the collisions they cause.

A reduced alphabet replaces each residue with the letter of its class, so several
sequences can read as one string. That is the point -- it is what lets a search match
peptides whose residues have diverged -- and it is also the failure this benchmark has to
keep in view: under a two-class cysteine-versus-other reading, two unrelated toxins with
the same cysteine spacing are the same string, and no search can tell them apart.

Every alphabet here is fixed in advance, listed in `ALPHABETS`. There is no search over
alphabets: the two framework-aware ones are the ones kmerseek already has (cys2 and
cys_hp3), the two-class hydrophobic/polar one is kmerseek's usual default, and dayhoff6 is
the one richer alphabet this benchmark tries, declared before any result was looked at.

Sketch subsampling is off everywhere in this module. `kmer_set` returns every k-mer, which
is what `scaled=1` means in kmerseek, so a peptide of 25 residues contributes all of its
k-mers and none is dropped by a hash threshold. For peptides this short a sketch would
keep a handful of k-mers and the comparison would be mostly noise.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

from .shared import hp_conservation_utils

CANONICAL_AA = "ACDEFGHIKLMNPQRSTVWY"

#: The residue classes of each alphabet, as a list of strings, one per class.
#:
#: cys2 and cys_hp3 are kmerseek's two cysteine-aware alphabets, written here to match
#: `src/rust/alphabets.rs` on the branch the cys-frameworks runs use:
#:   cys2     cysteine, everything else. Two classes.
#:   cys_hp3  hp_pbotc_1st_ed2 with cysteine pulled out of the hydrophobic class into a
#:            class of its own, so the framework and the hydrophobic/polar pattern of the
#:            stretches between the cysteines are both in one string. Three classes.
#: The rest come from `notebooks/hp_conservation_utils.ALPHABET_CLUSTERS` unchanged.
_HP_PBOTC = hp_conservation_utils.ALPHABET_CLUSTERS["hp_pbotc_1st_ed2"]
assert _HP_PBOTC == ["ACFILMPVWY", "DEGHKNQRST"], (
    "hp_pbotc_1st_ed2 moved in hp_conservation_utils; cys_hp3 is built from it"
)

ALPHABETS: dict[str, list[str]] = {
    # Every residue its own class: no reduction at all, the comparison the others are
    # measured against.
    "protein20": list(CANONICAL_AA),
    "cys2": ["C", "ADEFGHIKLMNPQRSTVWY"],
    "cys_hp3": ["AFILMPVWY", "DEGHKNQRST", "C"],
    "hp_pbotc_1st_ed2": list(_HP_PBOTC),
    "dayhoff6": list(hp_conservation_utils.ALPHABET_CLUSTERS["dayhoff6"]),
}

#: The letter each alphabet writes for each class. Lower case for the reduced ones so an
#: encoded string is never mistaken for a sequence of residues; cysteine is always `c`
#: where it has a class of its own, which is how kmerseek writes it.
_LETTERS = {
    "protein20": CANONICAL_AA,
    "cys2": "cx",
    "cys_hp3": "hpc",
    "hp_pbotc_1st_ed2": "hp",
    "dayhoff6": "abcdef",
}


def _table(name: str) -> dict[str, str]:
    classes, letters = ALPHABETS[name], _LETTERS[name]
    assert len(classes) == len(letters), f"{name}: {len(classes)} classes, {len(letters)} letters"
    table = {}
    for letter, residues in zip(letters, classes):
        for residue in residues:
            table[residue] = letter
    missing = set(CANONICAL_AA) - set(table)
    assert not missing, f"{name} does not place {sorted(missing)}"
    return table


TABLES: dict[str, dict[str, str]] = {name: _table(name) for name in ALPHABETS}


def encode(seq: str, alphabet: str) -> str:
    """`seq` rewritten in `alphabet`.

    Raises KeyError on a residue outside the 20 canonical amino acids, rather than
    skipping it: a modified residue silently dropped would shorten the sequence and shift
    every k-mer after it.

    >>> encode("CKGKGAKC", "cys2")
    'cxxxxxxc'
    >>> encode("CKGKGAKC", "cys_hp3")
    'cpppphpc'
    """
    table = TABLES[alphabet]
    return "".join(table[residue] for residue in seq)


def kmer_set(seq: str, alphabet: str, k: int) -> set[str]:
    """Every distinct k-mer of `seq` in `alphabet`, with no subsampling.

    Empty when the sequence is shorter than k.

    >>> sorted(kmer_set("CKGKC", "cys2", 3))
    ['cxx', 'xxc', 'xxx']
    """
    encoded = encode(seq, alphabet)
    return {encoded[i : i + k] for i in range(len(encoded) - k + 1)}


def kmer_list(seq: str, alphabet: str, k: int) -> list[str]:
    """Every k-mer of `seq` in `alphabet` in order, repeats included.

    `kmer_set` is what the searches compare; this is what the feature counts report, so
    that a peptide whose k-mers mostly repeat can be told from one whose k-mers are all
    distinct.
    """
    encoded = encode(seq, alphabet)
    return [encoded[i : i + k] for i in range(len(encoded) - k + 1)]


def containment(query: set[str], target: set[str]) -> float:
    """The share of the query's k-mers the target holds, 0 when the query has none.

    Containment rather than Jaccard, because a mature peptide is often being compared with
    a precursor several times its length, and Jaccard would penalise the target for the
    residues processing removes.
    """
    if not query:
        return 0.0
    return len(query & target) / len(query)


@dataclass(frozen=True)
class CollisionReport:
    """What one alphabet and k-size lose, counted on a fixed set of peptides."""

    alphabet: str
    ksize: int
    n_peptides: int
    #: Peptides whose encoded string is identical to that of at least one other peptide.
    n_peptides_in_collision: int
    #: How many distinct encoded strings the peptides collapse onto.
    n_distinct_encodings: int
    #: The largest group of peptides sharing one encoded string.
    largest_collision_group: int
    #: Total k-mers with repeats, and distinct k-mers, summed over the peptides. The ratio
    #: says how much of a peptide's own sequence the alphabet makes repetitive.
    total_kmers: int
    distinct_kmers_per_peptide: int
    #: Distinct k-mers over the whole set, and how many peptides the commonest k-mer is in.
    n_distinct_kmers_overall: int
    commonest_kmer_peptide_count: int

    def as_row(self) -> dict:
        return {
            "alphabet": self.alphabet,
            "ksize": self.ksize,
            "n_peptides": self.n_peptides,
            "n_peptides_in_collision": self.n_peptides_in_collision,
            "share_peptides_in_collision": (
                self.n_peptides_in_collision / self.n_peptides if self.n_peptides else 0.0
            ),
            "n_distinct_encodings": self.n_distinct_encodings,
            "largest_collision_group": self.largest_collision_group,
            "total_kmers": self.total_kmers,
            "distinct_kmers_per_peptide": self.distinct_kmers_per_peptide,
            "n_distinct_kmers_overall": self.n_distinct_kmers_overall,
            "commonest_kmer_peptide_count": self.commonest_kmer_peptide_count,
        }


def collision_report(
    peptides: dict[str, str], alphabet: str, ksize: int
) -> CollisionReport:
    """Count the peptides one alphabet cannot tell apart, and the k-mers it reuses.

    `peptides` maps an identifier to a sequence. Two peptides collide when their whole
    encoded strings are equal, which is the strongest form of the failure: no k-mer
    comparison at any k can separate them.

    >>> report = collision_report({"a": "CKGKC", "b": "CRSRC", "c": "CKGKCA"}, "cys2", 3)
    >>> report.n_peptides_in_collision, report.n_distinct_encodings
    (2, 2)
    """
    by_encoding: dict[str, list[str]] = defaultdict(list)
    total_kmers = 0
    distinct_kmers = 0
    kmer_peptides: Counter[str] = Counter()
    for name, seq in peptides.items():
        by_encoding[encode(seq, alphabet)].append(name)
        kmers = kmer_list(seq, alphabet, ksize)
        total_kmers += len(kmers)
        distinct = set(kmers)
        distinct_kmers += len(distinct)
        kmer_peptides.update(distinct)
    groups = [names for names in by_encoding.values() if len(names) > 1]
    commonest = kmer_peptides.most_common(1)
    return CollisionReport(
        alphabet=alphabet,
        ksize=ksize,
        n_peptides=len(peptides),
        n_peptides_in_collision=sum(len(names) for names in groups),
        n_distinct_encodings=len(by_encoding),
        largest_collision_group=max((len(names) for names in groups), default=0),
        total_kmers=total_kmers,
        distinct_kmers_per_peptide=distinct_kmers,
        n_distinct_kmers_overall=len(kmer_peptides),
        commonest_kmer_peptide_count=commonest[0][1] if commonest else 0,
    )


def collision_groups(peptides: dict[str, str], alphabet: str) -> list[list[str]]:
    """Every set of two or more peptides `alphabet` writes as the same string.

    Sorted largest first, then by name, so the list is the same on every run.
    """
    by_encoding: dict[str, list[str]] = defaultdict(list)
    for name, seq in peptides.items():
        by_encoding[encode(seq, alphabet)].append(name)
    groups = [sorted(names) for names in by_encoding.values() if len(names) > 1]
    return sorted(groups, key=lambda names: (-len(names), names))
