"""Construct-check kill test (notebook 355): the comparison workflow, the reduced-alphabet
check, the controls, and the fixed pass rule.

The question: when a protein-production facility receives back a construct whose sequenced
and translated protein does not match what was ordered, does a reduced-alphabet kmerseek
check find the unexpected piece of sequence (a tag, a linker, another known insert) when
exact matching of the tag list and local alignment of received against intended do not?

The 200 historical facility submissions this is written for are not in the repository and
were never available to the analysis, so no number in this module has been run on real
submissions. Everything here is written and unit-checked against the frozen tag list in
``data/kill_tests/construct_check/`` so that it runs the day the submissions arrive.

Three methods, all producing the same :class:`Flag` records so they can be compared:

- :func:`exact_tag_flags`: the tag and linker sequences looked for as exact substrings of
  the received protein, in the full 20-letter amino-acid alphabet. This is what a facility
  does already.
- :func:`alignment_flags`: local alignment of received against intended, with the stretches
  of received that the alignment does not cover reported as inserts. Also what a facility
  does already.
- :func:`reduced_kmer_flags`: the piece under test. The received protein and every tag are
  written in a reduced alphabet (for example hydrophobic/polar), and a flag is raised where
  they share a run of identical reduced k-mers. This is the seeding step kmerseek uses, run
  here in Python so the mechanics check needs no index.

A DNA-level check (VecScreen or the facility's own) is held by :func:`dna_check_flags`,
which reads the facility's own call file. It is part of the comparison, not withheld.
"""

from __future__ import annotations

import dataclasses
import itertools
import json
import math
import pathlib
import random
import statistics
from collections.abc import Iterable, Sequence

from ortholog_analysis_utils import AA_FREQUENCIES, ALPHABET_AA_GROUPS

FROZEN_DIR = (
    pathlib.Path(__file__).resolve().parents[1]
    / "data"
    / "kill_tests"
    / ("construct_check")
)

# Columns every submission record must have. The loader refuses a file that is missing one,
# so a half-filled export fails loudly instead of quietly shrinking the test.
SUBMISSION_COLUMNS = {
    "submission_id": "facility identifier for this submission, unique in the file",
    "intended_protein": "the protein sequence that was ordered, one-letter code, no gaps",
    "received_protein": "the sequenced and translated protein that came back, one-letter "
    "code, no gaps",
    "discrepancy_class": "one of: clean, unexpected_insert, point_change, truncation, "
    "frameshift, other",
    "resolved_insert_name": "for unexpected_insert, the name of the insert as resolved by "
    "the facility; empty otherwise",
    "resolved_insert_sequence": "for unexpected_insert, the inserted protein sequence as "
    "resolved; empty otherwise",
    "resolved_insert_start": "1-based first residue of the insert in received_protein; "
    "empty otherwise",
    "resolved_insert_end": "1-based last residue of the insert in received_protein, "
    "inclusive; empty otherwise",
    "insert_was_intended": "true when the insert is part of the order (a tag the customer "
    "asked for), false when it is a mistake",
    "split": "dev or test, assigned by intended protein, not by submission",
}
OPTIONAL_SUBMISSION_COLUMNS = {
    "received_dna": "the sequenced DNA, if the facility keeps it; empty otherwise",
    "batch_id": "the review batch this submission was reviewed in",
    "review_minutes": "minutes the facility recorded reviewing this batch",
}

DISCREPANCY_CLASSES = {
    "clean",
    "unexpected_insert",
    "point_change",
    "truncation",
    "frameshift",
    "other",
}


@dataclasses.dataclass(frozen=True)
class Flag:
    """One region of a received protein that a method says needs a human to look at it.

    Coordinates are 1-based and inclusive, counted along ``received_protein``, so they can
    be printed next to the residues without conversion.
    """

    submission_id: str
    method: str
    received_start: int
    received_end: int
    insert_name: str | None
    evidence: str

    def overlaps(self, start: int, end: int) -> int:
        """Number of residues this flag shares with the 1-based inclusive range given."""
        return max(0, min(self.received_end, end) - max(self.received_start, start) + 1)


@dataclasses.dataclass(frozen=True)
class TagSequence:
    """One entry of the frozen tag and linker list."""

    name: str
    category: str
    sequence: str
    source_kind: str
    source_accession: str
    source_coords_1based: str
    source_id: str
    provenance: str


def load_tag_list(path: pathlib.Path | None = None) -> list[TagSequence]:
    """Read the frozen tag and linker list."""
    path = path or FROZEN_DIR / "tags_linkers.tsv"
    lines = path.read_text().strip().split("\n")
    header = lines[0].split("\t")
    tags = []
    for line in lines[1:]:
        values = dict(zip(header, line.split("\t")))
        tags.append(
            TagSequence(
                name=values["name"],
                category=values["category"],
                sequence=values["sequence"],
                source_kind=values["source_kind"],
                source_accession=values["source_accession"],
                source_coords_1based=values["source_coords_1based"],
                source_id=values["source_id"],
                provenance=values["provenance"],
            )
        )
    return tags


def load_submissions(path: pathlib.Path) -> list[dict[str, str]]:
    """Read a submissions file and check it against the contract.

    Raises ``ValueError`` listing everything wrong, rather than returning a file that only
    looks right. A tab-separated file is expected, with the columns of
    :data:`SUBMISSION_COLUMNS` in any order; the columns of
    :data:`OPTIONAL_SUBMISSION_COLUMNS` may also be present.
    """
    # Only the trailing newline is stripped, not trailing tabs: ``strip()`` would remove the
    # empty last field of a row whose final column is blank, and that field would then go
    # missing instead of being reported.
    lines = path.read_text().rstrip("\n").split("\n")
    header = lines[0].split("\t")
    problems: list[str] = []
    missing = sorted(set(SUBMISSION_COLUMNS) - set(header))
    if missing:
        problems.append(f"missing required columns: {', '.join(missing)}")
    unknown = sorted(
        set(header) - set(SUBMISSION_COLUMNS) - set(OPTIONAL_SUBMISSION_COLUMNS)
    )
    if unknown:
        problems.append(f"columns not in the contract: {', '.join(unknown)}")
    repeated = sorted({name for name in header if header.count(name) > 1})
    if repeated:
        # dict(zip(...)) would keep only the last of a repeated column and lose the rest.
        problems.append(f"columns appearing more than once: {', '.join(repeated)}")
    if len(lines) < 2:
        problems.append(
            "the file has a header and no submissions. An empty export is refused rather "
            "than returned as zero rows, which would shrink the test to nothing"
        )
    if problems:
        raise ValueError(f"{path}: " + "; ".join(problems))

    # Every row must have exactly the header's fields. Without this a short row silently
    # becomes a dict missing a key, and the first use of that key raises KeyError from
    # somewhere else instead of being reported here.
    for index, line in enumerate(lines[1:], start=2):
        fields = line.split("\t")
        if len(fields) != len(header):
            problems.append(
                f"{path} line {index}: {len(fields)} fields, the header has {len(header)}"
            )
    if problems:
        raise ValueError("\n".join(problems))

    rows = [dict(zip(header, line.split("\t"))) for line in lines[1:]]
    seen: set[str] = set()
    for index, row in enumerate(rows, start=2):
        where = f"{path} line {index}"
        if row["submission_id"] in seen:
            problems.append(f"{where}: submission_id {row['submission_id']} repeats")
        seen.add(row["submission_id"])
        if row["discrepancy_class"] not in DISCREPANCY_CLASSES:
            problems.append(
                f"{where}: discrepancy_class {row['discrepancy_class']!r} is not one of "
                f"{sorted(DISCREPANCY_CLASSES)}"
            )
        if row["split"] not in {"dev", "test"}:
            problems.append(f"{where}: split must be dev or test, got {row['split']!r}")
        for column in ("intended_protein", "received_protein"):
            sequence = row[column]
            if not sequence:
                problems.append(f"{where}: {column} is empty")
            elif set(sequence) - set(AA_FREQUENCIES) - {"X", "*"}:
                bad = sorted(set(sequence) - set(AA_FREQUENCIES) - {"X", "*"})
                problems.append(f"{where}: {column} has non-residue letters {bad}")
        if row["discrepancy_class"] == "unexpected_insert":
            for column in (
                "resolved_insert_name",
                "resolved_insert_sequence",
                "resolved_insert_start",
                "resolved_insert_end",
            ):
                if not row[column]:
                    problems.append(f"{where}: {column} is required for an insert")
            if row["resolved_insert_sequence"] and row["resolved_insert_start"]:
                try:
                    start = int(row["resolved_insert_start"])
                    end = int(row["resolved_insert_end"])
                except ValueError:
                    problems.append(
                        f"{where}: resolved_insert_start and resolved_insert_end must be "
                        f"whole numbers, got {row['resolved_insert_start']!r} and "
                        f"{row['resolved_insert_end']!r}"
                    )
                else:
                    observed = row["received_protein"][start - 1 : end]
                    if observed != row["resolved_insert_sequence"]:
                        problems.append(
                            f"{where}: received_protein[{start}:{end}] is {observed!r}, "
                            "which is not the resolved insert "
                            f"{row['resolved_insert_sequence']!r}"
                        )
            if row["insert_was_intended"].lower() not in {"true", "false"}:
                problems.append(
                    f"{where}: insert_was_intended must be true or false, got "
                    f"{row['insert_was_intended']!r}"
                )

    # Contract rule: every submission sharing an intended protein is in the same split.
    # Without this the alphabet and k are chosen partly on proteins the test then scores,
    # and the test split is no longer held out. This was documented from the start and is
    # checked here since 2026-10-20, when an adversarial pass found the loader accepting a
    # file that broke it.
    splits_per_protein: dict[str, set[str]] = {}
    for row in rows:
        splits_per_protein.setdefault(row["intended_protein"], set()).add(row["split"])
    for protein, splits in splits_per_protein.items():
        if len(splits) > 1:
            ids = sorted(
                row["submission_id"]
                for row in rows
                if row["intended_protein"] == protein
            )
            problems.append(
                f"{path}: one intended protein ({len(protein)} aa, starting {protein[:20]!r}) "
                f"appears in both splits ({', '.join(sorted(splits))}), in submissions "
                f"{', '.join(ids)}. Split by intended protein, not by submission."
            )

    if problems:
        raise ValueError("\n".join(problems))
    return rows


def encode(sequence: str, alphabet: str) -> str:
    """Rewrite *sequence* in a reduced alphabet from :data:`ALPHABET_AA_GROUPS`.

    A residue the alphabet does not cover (``X``, a stop) becomes ``?``, which never
    matches, so an unknown residue cannot create a match it did not earn.
    """
    groups = ALPHABET_AA_GROUPS[alphabet]
    lookup = {
        amino_acid: symbol
        for symbol, amino_acids in groups.items()
        for amino_acid in amino_acids
    }
    return "".join(lookup.get(residue, "?") for residue in sequence)


def match_line(first: str, second: str) -> str:
    """The middle line of a residue-by-residue printout: ``|`` where the two agree."""
    return "".join("|" if a == b else " " for a, b in zip(first, second, strict=False))


def format_flagged_region(
    intended: str,
    received: str,
    start: int,
    end: int,
    alphabet: str | None = None,
    context: int = 10,
    line_width: int = 60,
) -> str:
    """Print the intended and received residues at a flagged region.

    *start* and *end* are 1-based inclusive coordinates along *received*. The two sequences
    are shown at the same offsets, with the match line between them and the count of
    identical positions, because a flag without its residues is not a result. When
    *alphabet* is given, the reduced-alphabet strings are printed underneath with their own
    match line.

    Both sequences are read at the same coordinates, so when the flagged region is an
    insert, everything after it on the intended line is the intended protein shifted left by
    the length of the insert. That is why the match line goes dark after the insert, and it
    is the reason the identical count is reported for the window shown rather than as an
    identity of the whole construct.
    """
    first = max(1, start - context)
    last = min(len(received), end + context)
    received_window = received[first - 1 : last]
    intended_window = intended[first - 1 : min(len(intended), last)]
    identical = sum(
        1 for a, b in zip(intended_window, received_window, strict=False) if a == b
    )
    compared = min(len(intended_window), len(received_window))

    lines = [
        f"flagged region: received {start}-{end} ({end - start + 1} aa), "
        f"shown with {context} aa of context on each side",
        f"{identical} of {compared} identical over the window shown",
    ]
    for offset in range(0, max(len(intended_window), len(received_window)), line_width):
        intended_chunk = intended_window[offset : offset + line_width]
        received_chunk = received_window[offset : offset + line_width]
        position = first + offset
        lines += [
            f"  intended {position:>6}  {intended_chunk}",
            f"  {'':>15}  {match_line(intended_chunk, received_chunk)}",
            f"  received {position:>6}  {received_chunk}",
        ]
        if alphabet is not None:
            intended_reduced = encode(intended_chunk, alphabet)
            received_reduced = encode(received_chunk, alphabet)
            lines += [
                f"  {alphabet} intended   {intended_reduced}",
                f"  {'':>15}  {match_line(intended_reduced, received_reduced)}",
                f"  {alphabet} received   {received_reduced}",
            ]
        marker = "".join(
            "^" if start <= position + index <= end else " "
            for index in range(len(received_chunk))
        )
        lines.append(f"  {'insert':>15}  {marker}")
    return "\n".join(lines)


def exact_tag_flags(
    submission_id: str, received: str, tags: Sequence[TagSequence]
) -> list[Flag]:
    """Flag every exact full-alphabet occurrence of a tag or linker in *received*."""
    flags = []
    for tag in tags:
        start = received.find(tag.sequence)
        while start >= 0:
            flags.append(
                Flag(
                    submission_id=submission_id,
                    method="exact tag match (protein20)",
                    received_start=start + 1,
                    received_end=start + len(tag.sequence),
                    insert_name=tag.name,
                    evidence=f"exact match to {tag.name} ({len(tag.sequence)} aa)",
                )
            )
            start = received.find(tag.sequence, start + 1)
    return flags


def _aligner():
    """A local BLOSUM62 aligner, built once per call to keep the import optional."""
    from Bio import Align
    from Bio.Align import substitution_matrices

    aligner = Align.PairwiseAligner()
    aligner.mode = "local"
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -11
    aligner.extend_gap_score = -1
    return aligner


def alignment_flags(
    submission_id: str, intended: str, received: str, min_insert_aa: int = 3
) -> list[Flag]:
    """Flag stretches of *received* that a local alignment to *intended* does not cover.

    Local alignment of received against intended is the facility's other existing check. A
    run of at least *min_insert_aa* residues of received left outside the aligned block, or
    inside it opposite a gap, is reported as an insert.
    """
    aligner = _aligner()
    try:
        alignment = next(iter(aligner.align(intended, received)))
    except (StopIteration, ValueError):
        return [
            Flag(
                submission_id=submission_id,
                method="local alignment (BLOSUM62)",
                received_start=1,
                received_end=len(received),
                insert_name=None,
                evidence="no local alignment to the intended protein",
            )
        ]
    covered = [False] * len(received)
    for (_, _), (received_start, received_end) in zip(
        alignment.aligned[0], alignment.aligned[1], strict=True
    ):
        for index in range(received_start, received_end):
            covered[index] = True

    flags = []
    for is_covered, group in itertools.groupby(
        enumerate(covered), key=lambda pair: pair[1]
    ):
        if is_covered:
            continue
        positions = [index for index, _ in group]
        if len(positions) < min_insert_aa:
            continue
        flags.append(
            Flag(
                submission_id=submission_id,
                method="local alignment (BLOSUM62)",
                received_start=positions[0] + 1,
                received_end=positions[-1] + 1,
                insert_name=None,
                evidence=f"{len(positions)} aa of the received protein not covered by the "
                "local alignment to the intended protein",
            )
        )
    return flags


def dna_check_flags(
    submission_id: str, calls: Iterable[dict[str, str]], received_length: int
) -> list[Flag]:
    """Turn the facility's own DNA-level calls (VecScreen or equivalent) into flags.

    *calls* are the rows the facility exported for this submission, each with
    ``protein_start``, ``protein_end`` and ``hit_name`` in 1-based protein coordinates. The
    DNA check is part of the comparison workflow and is never left out to make the protein
    method look better; when the facility supplies no DNA for a submission, this returns an
    empty list and the notebook counts that submission as having no DNA check.
    """
    flags = []
    for call in calls:
        start = int(call["protein_start"])
        end = min(int(call["protein_end"]), received_length)
        flags.append(
            Flag(
                submission_id=submission_id,
                method="DNA check (facility)",
                received_start=start,
                received_end=end,
                insert_name=call.get("hit_name") or None,
                evidence=f"DNA-level hit {call.get('hit_name', 'unnamed')}",
            )
        )
    return flags


def reduced_kmer_flags(
    submission_id: str,
    received: str,
    tags: Sequence[TagSequence],
    alphabet: str,
    ksize: int,
    min_run_kmers: int = 1,
) -> list[Flag]:
    """Flag regions of *received* that share reduced-alphabet k-mers with a tag.

    This is kmerseek's seeding step: both sides are written in *alphabet*, and a region is
    flagged where at least *min_run_kmers* consecutive k-mers of the received protein are
    also k-mers of a tag. Consecutive matching k-mers are merged into one flag, so a 20 aa
    tag match is one flag and not 20.
    """
    flags = []
    received_reduced = encode(received, alphabet)
    for tag in tags:
        if len(tag.sequence) < ksize:
            continue
        tag_reduced = encode(tag.sequence, alphabet)
        tag_kmers = {
            tag_reduced[index : index + ksize]
            for index in range(len(tag_reduced) - ksize + 1)
        }
        hit_starts = [
            index
            for index in range(len(received_reduced) - ksize + 1)
            if received_reduced[index : index + ksize] in tag_kmers
        ]
        for _, group in itertools.groupby(
            enumerate(hit_starts), key=lambda pair: pair[1] - pair[0]
        ):
            run = [start for _, start in group]
            if len(run) < min_run_kmers:
                continue
            flags.append(
                Flag(
                    submission_id=submission_id,
                    method=f"kmerseek reduced alphabet ({alphabet}, k={ksize})",
                    received_start=run[0] + 1,
                    received_end=run[-1] + ksize,
                    insert_name=tag.name,
                    evidence=f"{len(run)} consecutive {alphabet} k-mers of length {ksize} "
                    f"shared with {tag.name}",
                )
            )
    return flags


def choose_alphabet_and_k(
    dev_rows: Sequence[dict[str, str]],
    tags: Sequence[TagSequence],
    alphabets: Sequence[str],
    ksizes: Sequence[int],
    false_flag_budget_per_100: float = 1.0,
) -> tuple[str, int, list[dict[str, float]]]:
    """Pick one alphabet and one k on the development split, by a rule fixed in advance.

    The rule, written before any submission was seen: among the alphabet and k pairs whose
    false flags on the development split's clean constructs are at or under
    *false_flag_budget_per_100* per 100 clean constructs, take the one with the highest
    recall of documented unexpected inserts. Ties break toward the larger k, then the
    alphabet with more letters, because both make a chance match less likely.

    Returns the chosen alphabet, the chosen k, and the full grid so the notebook can draw
    every pair and mark the pick rather than showing only the winner.
    """
    grid = []
    for alphabet, ksize in itertools.product(alphabets, ksizes):
        recall, false_per_100 = _score_on_split(dev_rows, tags, alphabet, ksize)
        if math.isnan(recall):
            raise ValueError(
                "the development split has no documented unintended inserts, so recall "
                "cannot be scored and no alphabet and k can be picked"
            )
        if math.isnan(false_per_100):
            raise ValueError(
                "the development split has no clean constructs, so false flags per 100 "
                "cannot be scored and no alphabet and k can be picked"
            )
        grid.append(
            {
                "alphabet": alphabet,
                "ksize": ksize,
                "recall": recall,
                "false_flags_per_100_clean": false_per_100,
                "n_letters": len(ALPHABET_AA_GROUPS[alphabet]),
            }
        )
    affordable = [
        row
        for row in grid
        if row["false_flags_per_100_clean"] <= false_flag_budget_per_100
    ]
    if not affordable:
        raise ValueError(
            "no alphabet and k pair stays inside the false-flag budget of "
            f"{false_flag_budget_per_100} per 100 clean constructs on the development "
            "split; under the fixed rule that is a fail, not a reason to raise the budget"
        )
    best = max(
        affordable,
        key=lambda row: (row["recall"], row["ksize"], row["n_letters"]),
    )
    return best["alphabet"], int(best["ksize"]), grid


def _score_on_split(
    rows: Sequence[dict[str, str]],
    tags: Sequence[TagSequence],
    alphabet: str,
    ksize: int,
) -> tuple[float, float]:
    """Recall of documented unexpected inserts, and false flags per 100 clean constructs."""
    documented = [
        row
        for row in rows
        if row["discrepancy_class"] == "unexpected_insert"
        and row["insert_was_intended"].lower() == "false"
    ]
    clean = [row for row in rows if row["discrepancy_class"] == "clean"]
    recovered = 0
    for row in documented:
        flags = reduced_kmer_flags(
            row["submission_id"], row["received_protein"], tags, alphabet, ksize
        )
        start = int(row["resolved_insert_start"])
        end = int(row["resolved_insert_end"])
        if any(flag.overlaps(start, end) > 0 for flag in flags):
            recovered += 1
    false_flags = sum(
        len(
            reduced_kmer_flags(
                row["submission_id"], row["received_protein"], tags, alphabet, ksize
            )
        )
        for row in clean
    )
    recall = recovered / len(documented) if documented else float("nan")
    per_100 = 100 * false_flags / len(clean) if clean else float("nan")
    return recall, per_100


# --- controls -------------------------------------------------------------------------


def shuffle_sequence(sequence: str, seed: int) -> str:
    """Shuffle *sequence*, keeping its composition and length."""
    letters = list(sequence)
    random.Random(seed).shuffle(letters)
    return "".join(letters)


def composition_matched_stretch(
    donor: str, length: int, target_composition: str, seed: int, n_tries: int = 200
) -> str:
    """Pick the stretch of *donor* of *length* whose composition is closest to the target.

    Closeness is the sum over the 20 amino acids of the absolute difference in frequency.
    Used to insert a natural stretch that looks like the insert it stands in for, so that a
    flag cannot be explained by composition alone.
    """
    if length > len(donor):
        raise ValueError(f"donor is {len(donor)} aa, cannot take {length} aa from it")
    wanted = _composition(target_composition)
    rng = random.Random(seed)
    starts = (
        list(range(0, len(donor) - length + 1))
        if len(donor) - length + 1 <= n_tries
        else [rng.randrange(0, len(donor) - length + 1) for _ in range(n_tries)]
    )
    best_start = min(
        starts,
        key=lambda start: _composition_distance(
            wanted, _composition(donor[start : start + length])
        ),
    )
    return donor[best_start : best_start + length]


def _composition(sequence: str) -> dict[str, float]:
    """Amino-acid frequencies of *sequence*, over all 20 residues so every key is present."""
    if not sequence:
        return {amino_acid: 0.0 for amino_acid in AA_FREQUENCIES}
    return {
        amino_acid: sequence.count(amino_acid) / len(sequence)
        for amino_acid in AA_FREQUENCIES
    }


def _composition_distance(first: dict[str, float], second: dict[str, float]) -> float:
    """Sum over the 20 amino acids of the absolute difference in frequency."""
    return sum(
        abs(first.get(amino_acid, 0.0) - second.get(amino_acid, 0.0))
        for amino_acid in AA_FREQUENCIES
    )


def insert_at_random_position(
    host: str, insert: str, seed: int, margin: int = 20
) -> tuple[str, int, int]:
    """Put *insert* at a random position in *host*, away from both ends.

    Returns the new sequence and the 1-based inclusive coordinates of the insert in it.
    *margin* keeps the insert away from the termini, where a tag would normally sit, so the
    control does not quietly become a test of terminal tags.
    """
    if len(host) < 2 * margin + 1:
        raise ValueError(f"host is {len(host)} aa, too short for a margin of {margin}")
    position = random.Random(seed).randrange(margin, len(host) - margin)
    return (
        host[:position] + insert + host[position:],
        position + 1,
        position + len(insert),
    )


# --- the fixed pass rule --------------------------------------------------------------

PASS_RULE = {
    "1_recall_of_documented_inserts": {
        "threshold": 0.95,
        "direction": "at least",
        "what": "fraction of documented unexpected inserts in the test split that the "
        "reduced-alphabet check flags, counting a flag as recovering an insert when it "
        "overlaps the resolved insert coordinates by at least one residue",
    },
    "2_false_flags_per_100_clean": {
        "threshold": 1.0,
        "direction": "at most",
        "what": "flags raised on held-out clean constructs, per 100 clean constructs",
    },
    "3_comparison_missed_and_found": {
        "threshold": 5,
        "direction": "at least",
        "what": "documented errors the reduced-alphabet check flags that exact tag "
        "matching, local alignment and the DNA check all miss",
    },
    "4_median_review_time_per_batch": {
        "threshold": 0.0,
        "direction": "no increase",
        "what": "change in the facility's recorded median review minutes per batch when "
        "the reduced-alphabet flags are added to the existing ones. Assessable only if the "
        "facility supplies review_minutes; a count of flagged regions is not a substitute "
        "and the threshold is then reported as not assessable, which is not a pass",
    },
}

# Two shares this close count as the same, for the "cannot tell an ordered insert from a
# mistaken one" drop clause. Fixed with the rest of the rule, before any result.
SEPARATION_TOLERANCE = 0.1

FAIL_CLAUSES = {
    "exact_already_catches": "exact tag matching alone recovers at least as many "
    "documented unexpected inserts as the reduced-alphabet check",
    "cannot_separate_intended": "flags cannot tell an insert the customer ordered from "
    "one that is a mistake: the share of intended inserts flagged is within "
    f"{SEPARATION_TOLERANCE} of the share of unintended inserts flagged",
    "needs_dna": "every documented error the reduced-alphabet check finds beyond the "
    "comparison workflow is also found by the DNA check, so the protein method adds "
    "nothing the DNA already gives",
}


def evaluate_pass_rule(
    test_rows: Sequence[dict[str, str]],
    flags_by_method: dict[str, list[Flag]],
    reduced_method: str,
    review_minutes_before: Sequence[float] | None = None,
    review_minutes_after: Sequence[float] | None = None,
) -> dict[str, object]:
    """Score the four fixed thresholds and the three fail clauses on the test split.

    Returns one entry per threshold with its number, whether it holds, and the counts
    behind it, plus the verdict. Nothing here depends on how the numbers come out: the
    thresholds are the ones in :data:`PASS_RULE`, written before any run.
    """
    documented = [
        row
        for row in test_rows
        if row["discrepancy_class"] == "unexpected_insert"
        and row["insert_was_intended"].lower() == "false"
    ]
    clean = [row for row in test_rows if row["discrepancy_class"] == "clean"]
    clean_ids = {row["submission_id"] for row in clean}
    reduced_flags = flags_by_method[reduced_method]
    comparison_methods = [
        method for method in flags_by_method if method != reduced_method
    ]

    def recovers(method: str, row: dict[str, str]) -> bool:
        start = int(row["resolved_insert_start"])
        end = int(row["resolved_insert_end"])
        return any(
            flag.submission_id == row["submission_id"] and flag.overlaps(start, end) > 0
            for flag in flags_by_method[method]
        )

    recovered = [row for row in documented if recovers(reduced_method, row)]
    recall = len(recovered) / len(documented) if documented else float("nan")

    false_flags = [flag for flag in reduced_flags if flag.submission_id in clean_ids]
    per_100 = 100 * len(false_flags) / len(clean) if clean else float("nan")

    comparison_missed = [
        row
        for row in recovered
        if not any(recovers(method, row) for method in comparison_methods)
    ]

    results: dict[str, object] = {
        "1_recall_of_documented_inserts": {
            "value": recall,
            "threshold": PASS_RULE["1_recall_of_documented_inserts"]["threshold"],
            "holds": bool(documented)
            and recall >= PASS_RULE["1_recall_of_documented_inserts"]["threshold"],
            "counts": f"{len(recovered)} of {len(documented)} documented unexpected "
            "inserts flagged",
        },
        "2_false_flags_per_100_clean": {
            "value": per_100,
            "threshold": PASS_RULE["2_false_flags_per_100_clean"]["threshold"],
            "holds": bool(clean)
            and per_100 <= PASS_RULE["2_false_flags_per_100_clean"]["threshold"],
            "counts": f"{len(false_flags)} flags on {len(clean)} clean constructs",
        },
        "3_comparison_missed_and_found": {
            "value": len(comparison_missed),
            "threshold": PASS_RULE["3_comparison_missed_and_found"]["threshold"],
            "holds": len(comparison_missed)
            >= PASS_RULE["3_comparison_missed_and_found"]["threshold"],
            "counts": "found only by the reduced alphabet: "
            + (
                ", ".join(row["submission_id"] for row in comparison_missed)
                if comparison_missed
                else "none"
            ),
        },
    }

    if review_minutes_before and review_minutes_after:
        before = statistics.median(review_minutes_before)
        after = statistics.median(review_minutes_after)
        results["4_median_review_time_per_batch"] = {
            "value": after - before,
            "threshold": 0.0,
            "holds": after <= before,
            "counts": f"median {before:.1f} min per batch before, {after:.1f} min after, "
            f"n={len(review_minutes_before)} and {len(review_minutes_after)} batches",
        }
    else:
        results["4_median_review_time_per_batch"] = {
            "value": None,
            "threshold": 0.0,
            "holds": False,
            "counts": "not assessable: the facility supplied no review_minutes. Under the "
            "fixed rule a threshold that cannot be measured does not hold.",
        }

    exact_method = next(
        (method for method in comparison_methods if method.startswith("exact")), None
    )
    exact_recovered = (
        sum(1 for row in documented if recovers(exact_method, row))
        if exact_method
        else 0
    )
    dna_method = next(
        (method for method in comparison_methods if method.startswith("DNA")), None
    )
    # Can a flag tell an insert the customer ordered from one nobody ordered? Measured as the
    # share of intended inserts flagged against the share of unintended ones. If the method
    # flags both at the same rate it is reporting "an insert is here", which the reviewer
    # already knew from the order form, and the drop clause fires. A tolerance of 0.1 counts
    # rates this close as the same; it is part of the frozen rule, not tuned to a result.
    intended_inserts = [
        row
        for row in test_rows
        if row["discrepancy_class"] == "unexpected_insert"
        and row["insert_was_intended"].lower() == "true"
    ]
    intended_flagged = sum(
        1 for row in intended_inserts if recovers(reduced_method, row)
    )
    share_intended = (
        intended_flagged / len(intended_inserts) if intended_inserts else None
    )
    share_unintended = recall if documented else None
    cannot_separate = (
        share_intended is not None
        and share_unintended is not None
        and abs(share_intended - share_unintended) <= SEPARATION_TOLERANCE
    )

    results["fail_clauses"] = {
        "exact_already_catches": bool(exact_method)
        and exact_recovered >= len(recovered),
        "needs_dna": bool(dna_method)
        and bool(comparison_missed)
        and all(recovers(dna_method, row) for row in comparison_missed),
        "cannot_separate_intended": cannot_separate,
    }
    # A clause that could not be measured is recorded as such. It does not read as "the
    # clause did not fire", because an unmeasured drop clause is not evidence of passing.
    results["fail_clauses_not_measured"] = {
        "exact_already_catches": not exact_method,
        "needs_dna": not dna_method,
        "cannot_separate_intended": not intended_inserts,
    }
    results["comparison_methods_missing"] = [
        label
        for label, prefix in (
            ("exact tag matching", "exact"),
            ("local alignment", "local alignment"),
            ("DNA check", "DNA"),
        )
        if not any(method.startswith(prefix) for method in comparison_methods)
    ]
    results["separation"] = {
        "share_of_intended_inserts_flagged": share_intended,
        "share_of_unintended_inserts_flagged": share_unintended,
        "n_intended_inserts": len(intended_inserts),
        "n_unintended_inserts": len(documented),
    }
    thresholds_hold = all(results[key]["holds"] for key in PASS_RULE)
    measured_failure = any(results["fail_clauses"].values())
    missing_controls = any(results["fail_clauses_not_measured"].values()) or bool(
        results["comparison_methods_missing"]
    )
    if measured_failure:
        results["verdict"] = "FAIL"
    elif missing_controls:
        results["verdict"] = "INSUFFICIENT_EVIDENCE"
    else:
        results["verdict"] = "PASS" if thresholds_hold else "FAIL"
    return results


def write_pass_rule(path: pathlib.Path) -> None:
    """Freeze the pass rule to a file, so the committed version can be diffed later."""
    path.write_text(
        json.dumps({"pass_rule": PASS_RULE, "fail_clauses": FAIL_CLAUSES}, indent=2)
        + "\n"
    )
