"""The four workflows, all writing the same flag format.

- ``exact``   full-alphabet exact search for dictionary inserts and their pieces
- ``align``   local alignment of intended against received, then naming by alignment
- ``dna``     translate the original DNA, compare to intended, name by alignment
- ``reduced`` the method under test: shared k-mer runs in a reduced amino-acid alphabet

A flag never corrects anything. It says where, what, whether the specification permits it
there, and how strong the match is.
"""

from __future__ import annotations

import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "notebooks"))

import numpy as np
from Bio import Align
from Bio.Align import substitution_matrices
from Bio.Seq import Seq

from dictionary import DictionaryEntry
from records import Construct

# Alphabet tables are the repo's copy of kmerseek's src/rust/alphabets.rs.
from hp_conservation_utils import ALPHABET_CLUSTERS, GAP, TABLES

#: Frozen method settings, from PLAN.md.
REDUCED_SETTINGS = (("hp_thomas_dill2", 20), ("dayhoff6", 10))

#: Shortest exact piece of an insert the full-alphabet search reports.
EXACT_PIECE = 8

#: An extra received stretch shorter than this is not worth naming.
MIN_EXTRA = 5

#: Naming an extra stretch against the dictionary: identity over the insert, and the
#: fraction of the insert covered.
NAME_MIN_IDENTITY = 0.6
NAME_MIN_COVERAGE = 0.8

#: A flag whose residues match the intended sequence this well is ambiguous: a natural
#: stretch explains them as well as an insert does.
AMBIGUOUS_IDENTITY = 0.9

OUTCOMES = ("unexpected", "permitted", "ambiguous")


@dataclass
class Flag:
    """One region a workflow asks a human to look at."""

    workflow: str
    construct_id: str
    version: str
    start: int  # 1-based, inclusive, on the received sequence
    end: int
    residues: str
    insert_name: str
    support: str
    permitted: bool
    outcome: str
    region_evalue: float | None
    setting: str
    notes: str = ""

    def as_row(self) -> dict[str, object]:
        row = asdict(self)
        row["region_evalue"] = "" if self.region_evalue is None else f"{self.region_evalue:.3g}"
        return row

    @property
    def span(self) -> tuple[int, int]:
        return self.start, self.end


def _aligner(mode: str) -> Align.PairwiseAligner:
    aligner = Align.PairwiseAligner()
    aligner.mode = mode
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -11
    aligner.extend_gap_score = -1
    return aligner


#: Built once: loading BLOSUM62 from disk for every call was the slowest step of a run.
_ALIGNERS = {mode: _aligner(mode) for mode in ("local", "global")}


def _safe(seq: str) -> str:
    """BLOSUM62 has no row for sequencing placeholders; map them to X."""
    return "".join(r if r in "ACDEFGHIKLMNPQRSTVWYX" else "X" for r in seq)


def _identity(a: str, b: str) -> float:
    """Best local identity of ``a`` within ``b``, as a fraction of len(a)."""
    if not a or not b:
        return 0.0
    aligner = _ALIGNERS["local"]
    try:
        alignment = aligner.align(_safe(b), _safe(a))[0]
    except (ValueError, IndexError):
        return 0.0
    target, query = alignment[0], alignment[1]
    same = sum(1 for t, q in zip(target, query) if t == q and t != "-")
    return same / len(a)


def name_against_dictionary(
    residues: str, dictionary: list[DictionaryEntry]
) -> tuple[str, str, float]:
    """Best dictionary name for an extra stretch, with the identity that earned it."""
    best = ("", "", 0.0)
    for entry in dictionary:
        identity = _identity(entry.insert_seq, residues)
        coverage = min(len(residues), len(entry.insert_seq)) / len(entry.insert_seq)
        if identity < NAME_MIN_IDENTITY or coverage < NAME_MIN_COVERAGE:
            continue
        if identity > best[2]:
            best = (
                entry.insert_name,
                f"local alignment to {entry.insert_name}, identity {identity:.2f}",
                identity,
            )
    if not best[0]:
        return ("unnamed", "no dictionary insert reached the naming cutoff", 0.0)
    return best


def _classify(
    construct: Construct,
    insert_name: str,
    start: int,
    end: int,
    residues: str,
    run_length: int | None,
    l_distinguish: int | None,
) -> tuple[bool, str, str]:
    """Permitted, outcome and the note explaining an ambiguous call."""
    permitted = construct.permits(insert_name, start, end)
    if permitted:
        return True, "permitted", "the specification lists this insert at this place"
    if (
        run_length is not None
        and l_distinguish is not None
        and run_length < l_distinguish
    ):
        return (
            False,
            "ambiguous",
            f"matched stretch is {run_length} residues, below the {l_distinguish} "
            "needed to tell an insert from a composition-matched natural stretch",
        )
    identity = _identity(residues, construct.intended_protein_seq)
    if identity >= AMBIGUOUS_IDENTITY:
        return (
            False,
            "ambiguous",
            f"these residues also match the intended sequence at identity "
            f"{identity:.2f}, so a natural stretch explains them too",
        )
    return False, "unexpected", ""


def _extra_stretches(intended: str, received: str) -> list[tuple[int, int, str]]:
    """Received stretches with no intended counterpart, as 1-based inclusive spans."""
    aligner = _ALIGNERS["global"]
    try:
        alignment = aligner.align(_safe(intended), _safe(received))[0]
    except (ValueError, IndexError):
        return []
    target, query = alignment[0], alignment[1]
    spans = []
    received_pos = 0
    run_start = None
    for t, q in zip(target, query):
        if q != "-":
            received_pos += 1
        if t == "-" and q != "-":
            if run_start is None:
                run_start = received_pos
        else:
            if run_start is not None:
                spans.append((run_start, received_pos - (0 if q == "-" else 1)))
                run_start = None
    if run_start is not None:
        spans.append((run_start, received_pos))
    out = []
    for start, end in spans:
        if end - start + 1 >= MIN_EXTRA:
            out.append((start, end, received[start - 1 : end]))
    return out


def run_exact(
    construct: Construct, dictionary: list[DictionaryEntry], l_distinguish: int | None
) -> list[Flag]:
    """Full-alphabet exact search: whole inserts, and every piece of at least 8 residues.

    Pieces are included so a tag carrying one substitution is still found. Without them
    this workflow would be weaker than anyone would actually run.
    """
    received = construct.received_protein_seq
    flags: list[Flag] = []
    claimed: set[tuple[str, int, int]] = set()
    for entry in dictionary:
        insert = entry.insert_seq
        lengths = [len(insert)] + list(range(len(insert) - 1, EXACT_PIECE - 1, -1))
        for length in lengths:
            if length > len(received):
                continue
            for offset in range(len(insert) - length + 1):
                piece = insert[offset : offset + length]
                position = received.find(piece)
                while position != -1:
                    start, end = position + 1, position + length
                    key = (entry.insert_name, start, end)
                    inside = any(
                        name == entry.insert_name and start >= s and end <= e
                        for name, s, e in claimed
                    )
                    if not inside and key not in claimed:
                        claimed.add(key)
                        residues = received[start - 1 : end]
                        permitted, outcome, note = _classify(
                            construct,
                            entry.insert_name,
                            start,
                            end,
                            residues,
                            length,
                            l_distinguish,
                        )
                        support = (
                            f"exact match of all {length} residues of {entry.insert_name}"
                            if length == len(insert)
                            else f"exact match of {length} of {len(insert)} residues of "
                            f"{entry.insert_name}, offset {offset + 1}"
                        )
                        flags.append(
                            Flag(
                                workflow="exact",
                                construct_id=construct.construct_id,
                                version=construct.version,
                                start=start,
                                end=end,
                                residues=residues,
                                insert_name=entry.insert_name,
                                support=support,
                                permitted=permitted,
                                outcome=outcome,
                                region_evalue=None,
                                setting="full alphabet, exact",
                                notes=note,
                            )
                        )
                    position = received.find(piece, position + 1)
    return flags


def run_align(
    construct: Construct, dictionary: list[DictionaryEntry], l_distinguish: int | None
) -> list[Flag]:
    """Local alignment of intended against received, then naming by alignment."""
    flags = []
    for start, end, residues in _extra_stretches(
        construct.intended_protein_seq, construct.received_protein_seq
    ):
        name, support, _ = name_against_dictionary(residues, dictionary)
        permitted, outcome, note = _classify(
            construct, name, start, end, residues, len(residues), l_distinguish
        )
        flags.append(
            Flag(
                workflow="align",
                construct_id=construct.construct_id,
                version=construct.version,
                start=start,
                end=end,
                residues=residues,
                insert_name=name,
                support=f"{len(residues)} residues with no intended counterpart; {support}",
                permitted=permitted,
                outcome=outcome,
                region_evalue=None,
                setting="BLOSUM62 local alignment",
                notes=note,
            )
        )
    return flags


def run_dna(
    construct: Construct, dictionary: list[DictionaryEntry], l_distinguish: int | None
) -> list[Flag]:
    """Translate the original DNA, then compare that translation to the intended protein."""
    if not construct.original_dna_seq:
        return []
    received = construct.received_protein_seq
    best = (0.0, "", 0)
    for frame in range(3):
        chunk = construct.original_dna_seq[frame:]
        chunk = chunk[: len(chunk) - len(chunk) % 3]
        if not chunk:
            continue
        protein = str(Seq(chunk).translate()).replace("*", "")
        score = _identity(received[: min(len(received), 60)], protein)
        if score > best[0]:
            best = (score, protein, frame + 1)
    if not best[1]:
        return []
    translation = best[1]
    flags = []
    for start, end, residues in _extra_stretches(
        construct.intended_protein_seq, translation
    ):
        # Report coordinates on the received sequence, which is what a reviewer reads.
        position = received.find(residues)
        if position == -1:
            r_start, r_end = start, end
            coordinate_note = "coordinates are on the DNA translation; not found verbatim in the received sequence"
        else:
            r_start, r_end = position + 1, position + len(residues)
            coordinate_note = ""
        name, support, _ = name_against_dictionary(residues, dictionary)
        permitted, outcome, note = _classify(
            construct, name, r_start, r_end, residues, len(residues), l_distinguish
        )
        flags.append(
            Flag(
                workflow="dna",
                construct_id=construct.construct_id,
                version=construct.version,
                start=r_start,
                end=r_end,
                residues=residues,
                insert_name=name,
                support=(
                    f"DNA frame {best[2]}; {len(residues)} translated residues with "
                    f"no intended counterpart; {support}"
                ),
                permitted=permitted,
                outcome=outcome,
                region_evalue=None,
                setting="three-frame translation, then alignment",
                notes="; ".join(x for x in (note, coordinate_note) if x),
            )
        )
    return flags


def encode(seq: str, alphabet: str) -> np.ndarray:
    """Amino acids to reduced-alphabet class indices. Unknown letters become GAP."""
    table = TABLES[alphabet]
    codes = np.frombuffer(seq.encode(), dtype=np.uint8)
    return table[codes]


def _letter_frequencies(encoded: np.ndarray, alphabet: str) -> np.ndarray:
    classes = len(ALPHABET_CLUSTERS[alphabet])
    kept = encoded[encoded != GAP]
    counts = np.bincount(kept, minlength=classes).astype(float)
    if counts.sum() == 0:
        return np.full(classes, 1.0 / classes)
    # Add one of each class so a letter missing from this construct is not given
    # probability zero, which would make every E-value zero.
    counts += 1.0
    return counts / counts.sum()


def region_evalue(
    run: np.ndarray, frequencies: np.ndarray, received_length: int, dictionary_length: int
) -> float:
    """Chance of an exact reduced-alphabet run this long somewhere in the search space.

    This is the probability of matching these reduced letters by chance, multiplied by the
    number of places the match could have started. It is not the probability that the
    construct has an error, and it says nothing about whether the specification permits
    the insert.
    """
    with np.errstate(divide="ignore"):
        log_p = float(np.sum(np.log(frequencies[run])))
    space = max(received_length, 1) * max(dictionary_length, 1)
    return float(space * math.exp(log_p))


def _diagonal_runs(
    received_codes: np.ndarray, insert_codes: np.ndarray, k: int
) -> list[tuple[int, int, int]]:
    """Maximal ungapped runs of shared k-mers, as (received_start, insert_start, length).

    Positions are 0-based; length counts residues, so a single shared k-mer gives k.
    """
    if len(received_codes) < k or len(insert_codes) < k:
        return []

    def kmers(codes: np.ndarray) -> dict[bytes, list[int]]:
        out: dict[bytes, list[int]] = {}
        for i in range(len(codes) - k + 1):
            window = codes[i : i + k]
            if GAP in window:
                continue
            out.setdefault(window.tobytes(), []).append(i)
        return out

    insert_index = kmers(insert_codes)
    hits: dict[int, list[int]] = {}
    for i in range(len(received_codes) - k + 1):
        window = received_codes[i : i + k]
        if GAP in window:
            continue
        for j in insert_index.get(window.tobytes(), ()):
            hits.setdefault(i - j, []).append(i)

    runs = []
    for diagonal, positions in hits.items():
        positions.sort()
        run_start = positions[0]
        previous = positions[0]
        for position in positions[1:]:
            if position == previous + 1:
                previous = position
                continue
            runs.append((run_start, run_start - diagonal, previous - run_start + k))
            run_start = previous = position
        runs.append((run_start, run_start - diagonal, previous - run_start + k))
    return runs


def run_reduced(
    construct: Construct, dictionary: list[DictionaryEntry], l_distinguish: int | None
) -> list[Flag]:
    """The method under test, at both frozen settings."""
    received = construct.received_protein_seq
    flags = []
    for alphabet, k in REDUCED_SETTINGS:
        received_codes = encode(received, alphabet)
        frequencies = _letter_frequencies(received_codes, alphabet)
        dictionary_length = sum(len(e.insert_seq) for e in dictionary)
        for entry in dictionary:
            insert_codes = encode(entry.insert_seq, alphabet)
            for r_start, i_start, length in _diagonal_runs(
                received_codes, insert_codes, k
            ):
                start, end = r_start + 1, r_start + length
                residues = received[r_start : r_start + length]
                run = received_codes[r_start : r_start + length]
                evalue = region_evalue(
                    run, frequencies, len(received), dictionary_length
                )
                permitted, outcome, note = _classify(
                    construct,
                    entry.insert_name,
                    start,
                    end,
                    residues,
                    length,
                    l_distinguish,
                )
                flags.append(
                    Flag(
                        workflow="reduced",
                        construct_id=construct.construct_id,
                        version=construct.version,
                        start=start,
                        end=end,
                        residues=residues,
                        insert_name=entry.insert_name,
                        support=(
                            f"{length} residues matching {entry.insert_name} from its "
                            f"residue {i_start + 1} in {alphabet}, where "
                            f"{len(ALPHABET_CLUSTERS[alphabet])} letters stand for all 20 "
                            "amino acids"
                        ),
                        permitted=permitted,
                        outcome=outcome,
                        region_evalue=evalue,
                        setting=f"{alphabet}, k={k}",
                        notes=note,
                    )
                )
    return flags


WORKFLOWS = {
    "exact": run_exact,
    "align": run_align,
    "dna": run_dna,
    "reduced": run_reduced,
}

#: The workflows whose union is the comparison the method must beat.
COMPARISON = ("exact", "align", "dna")
