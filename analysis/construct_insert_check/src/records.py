"""Load and validate construct submission records.

The five input files are described in ``templates/README.md``. Everything here refuses
bad input loudly instead of guessing, because a silent coordinate or status error would
change the pass-rule counts.
"""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")
#: Letters a sequencing read may carry that are not one of the 20 amino acids.
TOLERATED_RESIDUES = set("BXZJUO*")
DNA_LETTERS = set("ACGTN")

STATUSES = ("clean", "error", "unresolved")
DECISIONS = ("unexpected_insert", "intended", "unresolved")

CONSTRUCT_COLUMNS = (
    "construct_id",
    "version",
    "group_id",
    "submitted_date",
    "status",
    "intended_protein_seq",
    "received_protein_seq",
    "received_seq_provenance",
    "original_dna_seq",
    "dna_provenance",
)
EXPECTED_INSERT_COLUMNS = (
    "construct_id",
    "version",
    "insert_name",
    "insert_seq",
    "expected_start",
    "expected_end",
)
EVENT_COLUMNS = (
    "construct_id",
    "version",
    "event_id",
    "insert_name",
    "insert_seq",
    "start",
    "end",
    "decision",
    "evidence",
    "decided_date",
)
DICTIONARY_COLUMNS = ("insert_name", "insert_seq", "category", "source", "frozen_date")
REVIEW_COLUMNS = (
    "batch_id",
    "workflow",
    "reviewer_id",
    "construct_count",
    "session_start",
    "session_end",
    "minutes",
    "source",
)

FILES = {
    "constructs.csv": CONSTRUCT_COLUMNS,
    "expected_inserts.csv": EXPECTED_INSERT_COLUMNS,
    "discrepancy_events.csv": EVENT_COLUMNS,
    "insert_dictionary.csv": DICTIONARY_COLUMNS,
    "review_sessions.csv": REVIEW_COLUMNS,
}


class ImportError_(ValueError):
    """An input file breaks the contract in ``templates/README.md``."""


@dataclass(frozen=True)
class ExpectedInsert:
    """An insert the specification permits, optionally pinned to coordinates."""

    insert_name: str
    insert_seq: str
    start: int | None
    end: int | None

    def permits(self, flag_start: int, flag_end: int) -> bool:
        """Does the specification allow this insert over ``flag_start..flag_end``?"""
        if self.start is None or self.end is None:
            return True
        # Permitted when the flag sits inside the specified window, allowing a 3-residue
        # slack on each side for the boundary residues an alignment may shift.
        return flag_start >= self.start - 3 and flag_end <= self.end + 3


@dataclass(frozen=True)
class Event:
    """A historical discrepancy decision."""

    construct_id: str
    version: str
    event_id: str
    insert_name: str
    insert_seq: str
    start: int
    end: int
    decision: str
    evidence: str
    decided_date: date


@dataclass(frozen=True)
class DictionaryInsert:
    """A known insert the search may look for."""

    insert_name: str
    insert_seq: str
    category: str
    source: str
    frozen_date: date


@dataclass
class Construct:
    """One construct submission version."""

    construct_id: str
    version: str
    group_id: str
    submitted_date: date
    status: str
    intended_protein_seq: str
    received_protein_seq: str
    received_seq_provenance: str
    original_dna_seq: str
    dna_provenance: str
    expected_inserts: list[ExpectedInsert] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)

    @property
    def key(self) -> str:
        return f"{self.construct_id}/{self.version}"

    @property
    def documented_events(self) -> list[Event]:
        """Events a reviewer settled as an unexpected insert."""
        return [e for e in self.events if e.decision == "unexpected_insert"]

    @property
    def is_clean(self) -> bool:
        """Clean means status clean and no documented event. Unresolved is not clean."""
        return self.status == "clean" and not self.documented_events

    def permits(self, insert_name: str, start: int, end: int) -> bool:
        name = insert_name.lower()
        return any(
            e.insert_name.lower() == name and e.permits(start, end)
            for e in self.expected_inserts
        )


@dataclass
class Collection:
    """Every record read from one input directory."""

    constructs: list[Construct]
    dictionary: list[DictionaryInsert]
    review_sessions: list[dict[str, str]]
    checksums: dict[str, str]

    def by_status(self) -> dict[str, int]:
        counts = {s: 0 for s in STATUSES}
        for c in self.constructs:
            counts[c.status] += 1
        return counts

    @property
    def documented_event_count(self) -> int:
        return sum(len(c.documented_events) for c in self.constructs)

    @property
    def clean_count(self) -> int:
        return sum(1 for c in self.constructs if c.is_clean)

    @property
    def with_dna_count(self) -> int:
        return sum(1 for c in self.constructs if c.original_dna_seq)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _read(path: Path, columns: tuple[str, ...]) -> list[dict[str, str]]:
    if not path.exists():
        raise ImportError_(f"{path.name} is missing from {path.parent}")
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames or []
        if tuple(header) != columns:
            raise ImportError_(
                f"{path.name} header is {header}, expected {list(columns)}"
            )
        return [{k: (v or "").strip() for k, v in row.items()} for row in reader]


def _parse_date(value: str, where: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ImportError_(f"{where}: {value!r} is not a YYYY-MM-DD date") from exc


def _check_protein(seq: str, where: str) -> str:
    seq = seq.upper()
    if not seq:
        raise ImportError_(f"{where}: protein sequence is empty")
    bad = set(seq) - AMINO_ACIDS - TOLERATED_RESIDUES
    if bad:
        raise ImportError_(f"{where}: not amino acid letters: {sorted(bad)}")
    return seq


def _check_dna(seq: str, where: str) -> str:
    seq = seq.upper()
    bad = set(seq) - DNA_LETTERS
    if bad:
        raise ImportError_(f"{where}: not DNA letters: {sorted(bad)}")
    return seq


def _coordinate(value: str, where: str, length: int) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise ImportError_(f"{where}: {value!r} is not an integer coordinate") from exc
    if not 1 <= number <= length:
        raise ImportError_(
            f"{where}: coordinate {number} is outside 1..{length}; "
            "coordinates are 1-based and inclusive on the received sequence"
        )
    return number


def load(directory: str | Path) -> Collection:
    """Read and validate one input directory. Raises ``ImportError_`` on any problem."""
    directory = Path(directory)
    checksums = {}
    for name in FILES:
        path = directory / name
        if not path.exists():
            raise ImportError_(f"{name} is missing from {directory}")
        checksums[name] = sha256(path)

    rows = {name: _read(directory / name, cols) for name, cols in FILES.items()}

    constructs: dict[tuple[str, str], Construct] = {}
    for row in rows["constructs.csv"]:
        where = f"constructs.csv {row['construct_id']}/{row['version']}"
        if row["status"] not in STATUSES:
            raise ImportError_(f"{where}: status {row['status']!r} not in {STATUSES}")
        if not row["group_id"]:
            raise ImportError_(f"{where}: group_id is empty")
        if not row["received_seq_provenance"]:
            raise ImportError_(f"{where}: received_seq_provenance is empty")
        dna = _check_dna(row["original_dna_seq"], where)
        if dna and not row["dna_provenance"]:
            raise ImportError_(f"{where}: original_dna_seq is set but dna_provenance is empty")
        construct = Construct(
            construct_id=row["construct_id"],
            version=row["version"],
            group_id=row["group_id"],
            submitted_date=_parse_date(row["submitted_date"], where),
            status=row["status"],
            intended_protein_seq=_check_protein(row["intended_protein_seq"], where),
            received_protein_seq=_check_protein(row["received_protein_seq"], where),
            received_seq_provenance=row["received_seq_provenance"],
            original_dna_seq=dna,
            dna_provenance=row["dna_provenance"],
        )
        key = (construct.construct_id, construct.version)
        if key in constructs:
            raise ImportError_(f"{where}: duplicated construct_id/version row")
        constructs[key] = construct

    groups: dict[str, str] = {}
    for construct in constructs.values():
        seen = groups.setdefault(construct.construct_id, construct.group_id)
        if seen != construct.group_id:
            raise ImportError_(
                f"constructs.csv {construct.construct_id}: versions disagree on group_id "
                f"({seen!r} and {construct.group_id!r}); every version of a construct "
                "must share one group, or a split would separate them"
            )

    for row in rows["expected_inserts.csv"]:
        key = (row["construct_id"], row["version"])
        where = f"expected_inserts.csv {row['construct_id']}/{row['version']}"
        if key not in constructs:
            raise ImportError_(f"{where}: no such construct version in constructs.csv")
        construct = constructs[key]
        length = len(construct.received_protein_seq)
        start = end = None
        if row["expected_start"] or row["expected_end"]:
            if not (row["expected_start"] and row["expected_end"]):
                raise ImportError_(f"{where}: give both expected_start and expected_end, or neither")
            start = _coordinate(row["expected_start"], where, length)
            end = _coordinate(row["expected_end"], where, length)
            if start > end:
                raise ImportError_(f"{where}: expected_start {start} is after expected_end {end}")
        construct.expected_inserts.append(
            ExpectedInsert(
                insert_name=row["insert_name"],
                insert_seq=_check_protein(row["insert_seq"], where),
                start=start,
                end=end,
            )
        )

    event_ids: set[str] = set()
    for row in rows["discrepancy_events.csv"]:
        key = (row["construct_id"], row["version"])
        where = f"discrepancy_events.csv {row['event_id']}"
        if key not in constructs:
            raise ImportError_(f"{where}: no such construct version in constructs.csv")
        if row["decision"] not in DECISIONS:
            raise ImportError_(f"{where}: decision {row['decision']!r} not in {DECISIONS}")
        if row["event_id"] in event_ids:
            raise ImportError_(f"{where}: duplicated event_id")
        event_ids.add(row["event_id"])
        construct = constructs[key]
        length = len(construct.received_protein_seq)
        start = _coordinate(row["start"], where, length)
        end = _coordinate(row["end"], where, length)
        if start > end:
            raise ImportError_(f"{where}: start {start} is after end {end}")
        if row["decision"] == "unexpected_insert" and not row["evidence"]:
            raise ImportError_(f"{where}: an unexpected_insert decision needs evidence")
        construct.events.append(
            Event(
                construct_id=row["construct_id"],
                version=row["version"],
                event_id=row["event_id"],
                insert_name=row["insert_name"],
                insert_seq=_check_protein(row["insert_seq"], where),
                start=start,
                end=end,
                decision=row["decision"],
                evidence=row["evidence"],
                decided_date=_parse_date(row["decided_date"], where),
            )
        )

    for construct in constructs.values():
        if construct.status == "clean" and construct.documented_events:
            raise ImportError_(
                f"constructs.csv {construct.key}: status is clean but "
                f"{len(construct.documented_events)} unexpected_insert events are recorded"
            )

    dictionary = []
    names: set[str] = set()
    for row in rows["insert_dictionary.csv"]:
        where = f"insert_dictionary.csv {row['insert_name']}"
        if row["insert_name"].lower() in names:
            raise ImportError_(f"{where}: duplicated insert_name")
        names.add(row["insert_name"].lower())
        dictionary.append(
            DictionaryInsert(
                insert_name=row["insert_name"],
                insert_seq=_check_protein(row["insert_seq"], where),
                category=row["category"],
                source=row["source"],
                frozen_date=_parse_date(row["frozen_date"], where),
            )
        )

    review_sessions = []
    for row in rows["review_sessions.csv"]:
        where = f"review_sessions.csv {row['batch_id']}/{row['workflow']}"
        if not row["source"]:
            raise ImportError_(
                f"{where}: source is empty; review minutes must name the tool or log "
                "that recorded them"
            )
        try:
            minutes = float(row["minutes"])
        except ValueError as exc:
            raise ImportError_(f"{where}: minutes {row['minutes']!r} is not a number") from exc
        if minutes <= 0:
            raise ImportError_(f"{where}: minutes must be positive")
        row = dict(row, minutes=minutes)
        review_sessions.append(row)

    return Collection(
        constructs=list(constructs.values()),
        dictionary=dictionary,
        review_sessions=review_sessions,
        checksums=checksums,
    )
