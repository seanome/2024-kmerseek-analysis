"""Score the workflows against the frozen pass rule in PLAN.md.

Event recovery and construct false flags are counted separately and never pooled. An
ambiguous flag counts against the method and never for it.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field

from records import Collection, Construct, Event
from workflows import Flag

#: Frozen before evaluation: a flag matches an event at this interval overlap or more.
OVERLAP_THRESHOLD = 0.5

#: Frozen pass rule.
MIN_EVENT_RECALL = 0.95
MAX_CLEAN_FLAG_RATE = 0.01
MIN_EVENTS_BEATING_COMPARISON = 5

SHORT_INSERT = 15
HYDROPHOBIC = set("ACFILMVWY")
HYDROPHOBIC_FRACTION = 0.6
REPETITIVE_FRACTION = 0.4
REPETITIVE_PERIOD = 4


def overlap(a: tuple[int, int], b: tuple[int, int]) -> float:
    """Intersection over union of two 1-based inclusive intervals."""
    low = max(a[0], b[0])
    high = min(a[1], b[1])
    intersection = max(0, high - low + 1)
    union = (a[1] - a[0] + 1) + (b[1] - b[0] + 1) - intersection
    return intersection / union if union else 0.0


def matches(flag: Flag, event: Event) -> bool:
    """Frozen event matching: same insert name, interval overlap at least 0.5."""
    if flag.insert_name.lower() != event.insert_name.lower():
        return False
    return overlap(flag.span, (event.start, event.end)) >= OVERLAP_THRESHOLD


def insert_classes(seq: str) -> list[str]:
    """The reporting classes an insert belongs to. An insert may be in several."""
    classes = []
    if len(seq) < SHORT_INSERT:
        classes.append("short")
    if seq:
        top = max(seq.count(r) for r in set(seq)) / len(seq)
        periodic = any(
            period < len(seq)
            and all(seq[i] == seq[i % period] for i in range(len(seq)))
            for period in range(1, REPETITIVE_PERIOD + 1)
        )
        if top > REPETITIVE_FRACTION or periodic:
            classes.append("repetitive")
        if sum(1 for r in seq if r in HYDROPHOBIC) / len(seq) > HYDROPHOBIC_FRACTION:
            classes.append("hydrophobic")
    return classes or ["ordinary"]


@dataclass
class WorkflowScore:
    workflow: str
    events_total: int = 0
    events_recovered: int = 0
    recovered_event_ids: set[str] = field(default_factory=set)
    clean_constructs: int = 0
    clean_constructs_flagged: int = 0
    flags_total: int = 0
    flags_unexpected: int = 0
    flags_ambiguous: int = 0
    flags_permitted: int = 0
    by_class: dict[str, tuple[int, int]] = field(default_factory=dict)

    @property
    def event_recall(self) -> float | None:
        if self.events_total == 0:
            return None
        return self.events_recovered / self.events_total

    @property
    def clean_flag_rate(self) -> float | None:
        if self.clean_constructs == 0:
            return None
        return self.clean_constructs_flagged / self.clean_constructs


def _counts_as_flag(flag: Flag) -> bool:
    """A flag a reviewer must look at: unexpected, or ambiguous.

    An ambiguous flag costs reviewer attention, so it counts as a false flag on a clean
    construct. A permitted insert is not a flag at all.
    """
    return flag.outcome in ("unexpected", "ambiguous")


def score(
    collection: Collection, flags_by_workflow: dict[str, list[Flag]]
) -> dict[str, WorkflowScore]:
    """Per-workflow scores. ``flags_by_workflow`` may hold a union under its own name."""
    scores = {}
    for workflow, flags in flags_by_workflow.items():
        result = WorkflowScore(workflow=workflow)
        by_construct: dict[tuple[str, str], list[Flag]] = {}
        for flag in flags:
            by_construct.setdefault((flag.construct_id, flag.version), []).append(flag)
            result.flags_total += 1
            if flag.outcome == "unexpected":
                result.flags_unexpected += 1
            elif flag.outcome == "ambiguous":
                result.flags_ambiguous += 1
            else:
                result.flags_permitted += 1

        class_totals: dict[str, list[int]] = {}
        for construct in collection.constructs:
            here = [f for f in by_construct.get((construct.construct_id, construct.version), [])]
            if construct.is_clean:
                result.clean_constructs += 1
                if any(_counts_as_flag(f) for f in here):
                    result.clean_constructs_flagged += 1
            for event in construct.documented_events:
                result.events_total += 1
                found = any(
                    matches(f, event) and f.outcome == "unexpected" for f in here
                )
                if found:
                    result.events_recovered += 1
                    result.recovered_event_ids.add(event.event_id)
                for name in insert_classes(event.insert_seq):
                    totals = class_totals.setdefault(name, [0, 0])
                    totals[0] += 1
                    totals[1] += 1 if found else 0
        result.by_class = {k: (v[0], v[1]) for k, v in sorted(class_totals.items())}
        scores[workflow] = result
    return scores


@dataclass
class PassRule:
    event_recall_ok: bool | None
    clean_rate_ok: bool | None
    beats_comparison_ok: bool | None
    review_time_ok: bool | None
    events_beating_comparison: list[str]
    verdict: str
    reasons: list[str]


def review_time(collection: Collection) -> dict[str, float]:
    """Median recorded review minutes per batch, per workflow. Never machine runtime."""
    per_workflow: dict[str, list[float]] = {}
    for row in collection.review_sessions:
        per_workflow.setdefault(row["workflow"], []).append(float(row["minutes"]))
    return {k: statistics.median(v) for k, v in sorted(per_workflow.items())}


def apply_pass_rule(
    collection: Collection, scores: dict[str, WorkflowScore]
) -> PassRule:
    """The four frozen criteria. Any criterion without data makes the verdict unknown."""
    reasons = []
    method = scores.get("reduced")
    comparison = scores.get("comparison")
    if method is None or comparison is None:
        return PassRule(None, None, None, None, [], "INSUFFICIENT_EVIDENCE", ["no scores"])

    recall = method.event_recall
    event_recall_ok = None if recall is None else recall >= MIN_EVENT_RECALL
    if recall is None:
        reasons.append("criterion 1: no documented unexpected-insert events to recover")
    else:
        reasons.append(
            f"criterion 1: recovered {method.events_recovered} of {method.events_total} "
            f"events, {recall:.1%}, needs {MIN_EVENT_RECALL:.0%}"
        )

    clean_rate = method.clean_flag_rate
    clean_rate_ok = None if clean_rate is None else clean_rate <= MAX_CLEAN_FLAG_RATE
    if clean_rate is None:
        reasons.append("criterion 2: no clean constructs to flag")
    else:
        reasons.append(
            f"criterion 2: flagged {method.clean_constructs_flagged} of "
            f"{method.clean_constructs} clean constructs, {clean_rate:.1%}, "
            f"allows {MAX_CLEAN_FLAG_RATE:.0%}"
        )

    extra = sorted(method.recovered_event_ids - comparison.recovered_event_ids)
    beats_ok = (
        None
        if method.events_total == 0
        else len(extra) >= MIN_EVENTS_BEATING_COMPARISON
    )
    reasons.append(
        f"criterion 3: recovered {len(extra)} events the comparison missed, "
        f"needs {MIN_EVENTS_BEATING_COMPARISON}"
    )

    medians = review_time(collection)
    if "reduced" in medians and "comparison" in medians:
        review_ok = medians["reduced"] <= medians["comparison"]
        reasons.append(
            f"criterion 4: median recorded review {medians['reduced']:.1f} min per batch "
            f"against {medians['comparison']:.1f} min for the comparison"
        )
    else:
        review_ok = None
        reasons.append(
            "criterion 4: no recorded review sessions for both workflows, so review time "
            "cannot be compared; machine runtime is not review time"
        )

    checks = [event_recall_ok, clean_rate_ok, beats_ok, review_ok]
    if any(c is None for c in checks):
        verdict = "INSUFFICIENT_EVIDENCE"
    elif all(checks):
        verdict = "PASS"
    else:
        verdict = "FAIL"
    return PassRule(
        event_recall_ok=event_recall_ok,
        clean_rate_ok=clean_rate_ok,
        beats_comparison_ok=beats_ok,
        review_time_ok=review_ok,
        events_beating_comparison=extra,
        verdict=verdict,
        reasons=reasons,
    )


def paired_table(
    collection: Collection, scores: dict[str, WorkflowScore]
) -> list[dict[str, object]]:
    """One row per documented event, with a column per workflow saying caught or missed."""
    order = ["exact", "align", "dna", "comparison", "reduced"]
    rows = []
    for construct in collection.constructs:
        for event in construct.documented_events:
            row: dict[str, object] = {
                "construct_id": construct.construct_id,
                "version": construct.version,
                "event_id": event.event_id,
                "insert_name": event.insert_name,
                "start": event.start,
                "end": event.end,
                "insert_length": len(event.insert_seq),
                "insert_classes": "|".join(insert_classes(event.insert_seq)),
                "evidence": event.evidence,
            }
            for workflow in order:
                if workflow in scores:
                    row[workflow] = (
                        "caught"
                        if event.event_id in scores[workflow].recovered_event_ids
                        else "missed"
                    )
            rows.append(row)
    return rows


def enough_data(collection: Collection) -> tuple[bool, list[str]]:
    """Can the pass rule be assessed at all? Counts, and what is short."""
    problems = []
    if len(collection.constructs) < 200:
        problems.append(
            f"{len(collection.constructs)} construct versions, the plan asks for 200"
        )
    if collection.clean_count < 100:
        problems.append(
            f"{collection.clean_count} clean constructs; at most "
            f"{1 / max(collection.clean_count, 1):.1%} resolution on a 1% false-flag rate"
        )
    if collection.documented_event_count < 20:
        problems.append(
            f"{collection.documented_event_count} documented events; 95% recovery cannot "
            "be told from 90% with so few"
        )
    if not collection.review_sessions:
        problems.append("no recorded review sessions, so criterion 4 cannot be assessed")
    return (not problems), problems


def construct_report_row(construct: Construct, flags: list[Flag]) -> dict[str, object]:
    return {
        "construct_id": construct.construct_id,
        "version": construct.version,
        "group_id": construct.group_id,
        "status": construct.status,
        "received_seq_provenance": construct.received_seq_provenance,
        "has_dna": bool(construct.original_dna_seq),
        "documented_events": len(construct.documented_events),
        "flags_unexpected": sum(1 for f in flags if f.outcome == "unexpected"),
        "flags_ambiguous": sum(1 for f in flags if f.outcome == "ambiguous"),
        "flags_permitted": sum(1 for f in flags if f.outcome == "permitted"),
    }
