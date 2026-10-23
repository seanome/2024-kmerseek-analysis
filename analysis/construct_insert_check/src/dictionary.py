"""Build each construct's insert dictionary without letting its own answer leak in.

Two rules, both from ``PLAN.md``:

1. An insert may come from the frozen development collection, or from a decision dated
   strictly before this construct's submission date.
2. Nothing learned from any record sharing this construct's ``group_id`` may be used,
   whatever its date. Versions of one construct, and related constructs, share a group,
   so this is what stops an evaluation discrepancy from arriving through another copy of
   the same record.
"""

from __future__ import annotations

from dataclasses import dataclass

from records import Collection, Construct, DictionaryInsert


class LeakageError(RuntimeError):
    """A construct's own insert reached its dictionary. Always a hard failure."""


@dataclass(frozen=True)
class DictionaryEntry:
    insert_name: str
    insert_seq: str
    category: str
    origin: str
    #: The group a prior decision came from, so the group filter can be checked again
    #: here. Empty for the frozen development collection, which belongs to no group.
    from_group: str = ""


def build_for(construct: Construct, collection: Collection) -> list[DictionaryEntry]:
    """The inserts the search is allowed to look for in ``construct``."""
    entries: dict[str, DictionaryEntry] = {}

    for item in collection.dictionary:
        if item.frozen_date >= construct.submitted_date:
            continue
        entries[item.insert_name.lower()] = DictionaryEntry(
            insert_name=item.insert_name,
            insert_seq=item.insert_seq,
            category=item.category,
            origin=f"development collection frozen {item.frozen_date}",
            from_group="",
        )

    for other in collection.constructs:
        if other.group_id == construct.group_id:
            continue
        for event in other.events:
            if event.decision != "unexpected_insert":
                continue
            if event.decided_date >= construct.submitted_date:
                continue
            key = event.insert_name.lower()
            if key in entries:
                continue
            entries[key] = DictionaryEntry(
                insert_name=event.insert_name,
                insert_seq=event.insert_seq,
                category="from earlier decision",
                origin=f"decision {event.event_id} on {event.decided_date}",
                from_group=other.group_id,
            )

    for entry in entries.values():
        if entry.from_group == construct.group_id:
            raise LeakageError(
                f"{construct.key}: dictionary entry {entry.insert_name!r} "
                f"({entry.origin}) was learned from this construct's own group"
            )
    return sorted(entries.values(), key=lambda e: e.insert_name.lower())


def own_inserts_already_known(
    construct: Construct, dictionary: list[DictionaryEntry]
) -> int:
    """How many of this construct's event sequences the dictionary already held.

    Reported, not refused: the search is meant to look for known inserts. Criterion 3 is
    what tests whether knowing them is enough.
    """
    known = {e.insert_seq for e in dictionary}
    return sum(1 for e in construct.documented_events if e.insert_seq in known)


def leakage_report(collection: Collection) -> list[str]:
    """Every construct's dictionary, checked. Empty list means nothing leaked."""
    problems = []
    for construct in collection.constructs:
        try:
            build_for(construct, collection)
        except LeakageError as exc:
            problems.append(str(exc))
    return problems
