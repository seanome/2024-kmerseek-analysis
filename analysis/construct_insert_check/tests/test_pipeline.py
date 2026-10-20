"""Mechanics tests. Synthetic inputs only, so nothing here is evidence about constructs."""

from __future__ import annotations

import csv
import random
import sys
from datetime import date
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import controls  # noqa: E402
import dictionary as dictionary_module  # noqa: E402
import evaluate  # noqa: E402
import records  # noqa: E402
import workflows  # noqa: E402

TAG = "MSDSEVNQEAKPEVKPEVKPETHINLKVSDGSSEI"  # SUMO tail, 35 residues
CORE = (
    "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKR"
    "QTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFG"
)


def write_collection(
    directory: Path,
    constructs: list[dict],
    expected: list[dict] | None = None,
    events: list[dict] | None = None,
    dictionary_rows: list[dict] | None = None,
    review: list[dict] | None = None,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    tables = {
        "constructs.csv": (records.CONSTRUCT_COLUMNS, constructs),
        "expected_inserts.csv": (records.EXPECTED_INSERT_COLUMNS, expected or []),
        "discrepancy_events.csv": (records.EVENT_COLUMNS, events or []),
        "insert_dictionary.csv": (records.DICTIONARY_COLUMNS, dictionary_rows or []),
        "review_sessions.csv": (records.REVIEW_COLUMNS, review or []),
    }
    for name, (columns, rows) in tables.items():
        with (directory / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(columns))
            writer.writeheader()
            writer.writerows(rows)
    return directory


def construct_row(**overrides) -> dict:
    row = {
        "construct_id": "C1",
        "version": "v1",
        "group_id": "G1",
        "submitted_date": "2025-01-01",
        "status": "clean",
        "intended_protein_seq": CORE,
        "received_protein_seq": CORE,
        "received_seq_provenance": "sanger 2025-01-02",
        "original_dna_seq": "",
        "dna_provenance": "",
    }
    row.update(overrides)
    return row


def dictionary_row(**overrides) -> dict:
    row = {
        "insert_name": "SUMO tail",
        "insert_seq": TAG,
        "category": "solubility tag",
        "source": "published",
        "frozen_date": "2024-01-01",
    }
    row.update(overrides)
    return row


# --- validation -------------------------------------------------------------------


def test_loads_a_minimal_collection(tmp_path):
    directory = write_collection(tmp_path, [construct_row()], dictionary_rows=[dictionary_row()])
    collection = records.load(directory)
    assert len(collection.constructs) == 1
    assert set(collection.checksums) == set(records.FILES)


def test_missing_file_is_refused(tmp_path):
    directory = write_collection(tmp_path, [construct_row()])
    (directory / "review_sessions.csv").unlink()
    with pytest.raises(records.ImportError_, match="review_sessions.csv is missing"):
        records.load(directory)


def test_wrong_header_is_refused(tmp_path):
    directory = write_collection(tmp_path, [construct_row()])
    (directory / "constructs.csv").write_text("construct_id,version\nC1,v1\n")
    with pytest.raises(records.ImportError_, match="header is"):
        records.load(directory)


def test_unknown_status_is_refused(tmp_path):
    directory = write_collection(tmp_path, [construct_row(status="probably_fine")])
    with pytest.raises(records.ImportError_, match="status"):
        records.load(directory)


def test_clean_status_with_a_documented_event_is_refused(tmp_path):
    events = [
        {
            "construct_id": "C1",
            "version": "v1",
            "event_id": "E1",
            "insert_name": "SUMO tail",
            "insert_seq": TAG,
            "start": 5,
            "end": 10,
            "decision": "unexpected_insert",
            "evidence": "sanger trace",
            "decided_date": "2024-05-01",
        }
    ]
    directory = write_collection(tmp_path, [construct_row(status="clean")], events=events)
    with pytest.raises(records.ImportError_, match="status is clean"):
        records.load(directory)


def test_coordinate_outside_the_sequence_is_refused(tmp_path):
    events = [
        {
            "construct_id": "C1",
            "version": "v1",
            "event_id": "E1",
            "insert_name": "SUMO tail",
            "insert_seq": TAG,
            "start": 1,
            "end": 99999,
            "decision": "unexpected_insert",
            "evidence": "sanger trace",
            "decided_date": "2024-05-01",
        }
    ]
    directory = write_collection(tmp_path, [construct_row(status="error")], events=events)
    with pytest.raises(records.ImportError_, match="outside 1.."):
        records.load(directory)


def test_versions_disagreeing_on_group_is_refused(tmp_path):
    rows = [construct_row(), construct_row(version="v2", group_id="G2")]
    directory = write_collection(tmp_path, rows)
    with pytest.raises(records.ImportError_, match="disagree on group_id"):
        records.load(directory)


def test_review_minutes_need_a_source(tmp_path):
    review = [
        {
            "batch_id": "B1",
            "workflow": "reduced",
            "reviewer_id": "R1",
            "construct_count": 20,
            "session_start": "2025-02-01T09:00",
            "session_end": "2025-02-01T09:30",
            "minutes": 30,
            "source": "",
        }
    ]
    directory = write_collection(tmp_path, [construct_row()], review=review)
    with pytest.raises(records.ImportError_, match="source is empty"):
        records.load(directory)


def test_unresolved_is_not_a_clean_negative(tmp_path):
    events = [
        {
            "construct_id": "C1",
            "version": "v1",
            "event_id": "E1",
            "insert_name": "SUMO tail",
            "insert_seq": TAG,
            "start": 5,
            "end": 10,
            "decision": "unresolved",
            "evidence": "",
            "decided_date": "2024-05-01",
        }
    ]
    directory = write_collection(tmp_path, [construct_row(status="unresolved")], events=events)
    collection = records.load(directory)
    construct = collection.constructs[0]
    assert construct.is_clean is False
    assert construct.documented_events == []
    assert collection.clean_count == 0


# --- dictionary and leakage -------------------------------------------------------


def _collection_with_two_groups(tmp_path, decided="2024-05-01", same_group=False):
    rows = [
        construct_row(construct_id="C1", group_id="G1", status="error"),
        construct_row(
            construct_id="C2",
            group_id="G1" if same_group else "G2",
            status="error",
        ),
    ]
    events = [
        {
            "construct_id": "C2",
            "version": "v1",
            "event_id": "E2",
            "insert_name": "His6",
            "insert_seq": "HHHHHH",
            "start": 5,
            "end": 10,
            "decision": "unexpected_insert",
            "evidence": "sanger trace",
            "decided_date": decided,
        }
    ]
    directory = write_collection(tmp_path, rows, events=events)
    return records.load(directory)


def test_an_earlier_decision_from_another_group_enters_the_dictionary(tmp_path):
    collection = _collection_with_two_groups(tmp_path)
    entries = dictionary_module.build_for(collection.constructs[0], collection)
    assert [e.insert_name for e in entries] == ["His6"]


def test_a_decision_from_the_same_group_never_enters(tmp_path):
    collection = _collection_with_two_groups(tmp_path, same_group=True)
    entries = dictionary_module.build_for(collection.constructs[0], collection)
    assert entries == []


def test_a_later_decision_never_enters(tmp_path):
    collection = _collection_with_two_groups(tmp_path, decided="2026-01-01")
    entries = dictionary_module.build_for(collection.constructs[0], collection)
    assert entries == []


def test_a_development_insert_frozen_after_the_case_never_enters(tmp_path):
    directory = write_collection(
        tmp_path,
        [construct_row(submitted_date="2024-01-01")],
        dictionary_rows=[dictionary_row(frozen_date="2024-06-01")],
    )
    collection = records.load(directory)
    assert dictionary_module.build_for(collection.constructs[0], collection) == []


def test_own_inserts_already_known_is_counted_not_refused(tmp_path):
    events = [
        {
            "construct_id": "C1",
            "version": "v1",
            "event_id": "E1",
            "insert_name": "SUMO tail",
            "insert_seq": TAG,
            "start": 1,
            "end": len(TAG),
            "decision": "unexpected_insert",
            "evidence": "sanger trace",
            "decided_date": "2024-05-01",
        }
    ]
    directory = write_collection(
        tmp_path,
        [construct_row(status="error", received_protein_seq=TAG + CORE)],
        events=events,
        dictionary_rows=[dictionary_row()],
    )
    collection = records.load(directory)
    construct = collection.constructs[0]
    entries = dictionary_module.build_for(construct, collection)
    assert dictionary_module.own_inserts_already_known(construct, entries) == 1


# --- workflows --------------------------------------------------------------------


def _one_construct(tmp_path, received, expected=None, status="error", dna=""):
    directory = write_collection(
        tmp_path,
        [
            construct_row(
                status=status,
                received_protein_seq=received,
                original_dna_seq=dna,
                dna_provenance="synthetic" if dna else "",
            )
        ],
        expected=expected,
        dictionary_rows=[dictionary_row()],
    )
    collection = records.load(directory)
    construct = collection.constructs[0]
    return construct, dictionary_module.build_for(construct, collection)


def test_reduced_search_finds_a_planted_insert_at_the_right_coordinates(tmp_path):
    received = CORE[:60] + TAG + CORE[60:]
    construct, entries = _one_construct(tmp_path, received)
    flags = workflows.run_reduced(construct, entries, l_distinguish=None)
    assert flags, "the reduced search found nothing"
    for flag in flags:
        assert flag.insert_name == "SUMO tail"
        assert flag.start == 61
        assert flag.end == 60 + len(TAG)
        assert received[flag.start - 1 : flag.end] == flag.residues == TAG


def test_exact_search_finds_the_same_insert(tmp_path):
    received = CORE[:60] + TAG + CORE[60:]
    construct, entries = _one_construct(tmp_path, received)
    flags = workflows.run_exact(construct, entries, l_distinguish=None)
    whole = [f for f in flags if f.residues == TAG]
    assert whole and whole[0].start == 61


def test_alignment_finds_the_extra_stretch(tmp_path):
    received = CORE[:60] + TAG + CORE[60:]
    construct, entries = _one_construct(tmp_path, received)
    flags = workflows.run_align(construct, entries, l_distinguish=None)
    assert any(f.insert_name == "SUMO tail" and f.start == 61 for f in flags)


def test_dna_workflow_runs_only_with_dna(tmp_path):
    received = CORE[:60] + TAG + CORE[60:]
    construct, entries = _one_construct(tmp_path, received)
    assert workflows.run_dna(construct, entries, l_distinguish=None) == []


def test_dna_workflow_translates_and_finds_the_insert(tmp_path):
    received = CORE[:60] + TAG + CORE[60:]
    import synthetic

    dna = synthetic.back_translate(received)
    construct, entries = _one_construct(tmp_path, received, dna=dna)
    flags = workflows.run_dna(construct, entries, l_distinguish=None)
    assert any(f.residues == TAG for f in flags)


def test_a_specified_insert_at_its_place_is_permitted(tmp_path):
    received = TAG + CORE
    expected = [
        {
            "construct_id": "C1",
            "version": "v1",
            "insert_name": "SUMO tail",
            "insert_seq": TAG,
            "expected_start": 1,
            "expected_end": len(TAG),
        }
    ]
    construct, entries = _one_construct(tmp_path, received, expected=expected, status="clean")
    for flag in workflows.run_reduced(construct, entries, l_distinguish=None):
        assert flag.outcome == "permitted"
        assert flag.permitted is True


def test_the_same_insert_somewhere_else_is_not_permitted(tmp_path):
    received = CORE[:90] + TAG + CORE[90:]
    expected = [
        {
            "construct_id": "C1",
            "version": "v1",
            "insert_name": "SUMO tail",
            "insert_seq": TAG,
            "expected_start": 1,
            "expected_end": len(TAG),
        }
    ]
    construct, entries = _one_construct(tmp_path, received, expected=expected)
    flags = workflows.run_reduced(construct, entries, l_distinguish=None)
    assert flags and all(f.outcome == "unexpected" for f in flags)


def test_a_short_run_is_ambiguous_not_unexpected(tmp_path):
    received = CORE[:60] + TAG + CORE[60:]
    construct, entries = _one_construct(tmp_path, received)
    flags = workflows.run_reduced(construct, entries, l_distinguish=len(TAG) + 5)
    assert flags and all(f.outcome == "ambiguous" for f in flags)
    assert all("below the" in f.notes for f in flags)


def test_no_workflow_changes_a_sequence(tmp_path):
    received = CORE[:60] + TAG + CORE[60:]
    construct, entries = _one_construct(tmp_path, received)
    before = (construct.intended_protein_seq, construct.received_protein_seq)
    for workflow in workflows.WORKFLOWS.values():
        workflow(construct, entries, None)
    assert (construct.intended_protein_seq, construct.received_protein_seq) == before


def test_region_evalue_falls_as_the_run_grows(tmp_path):
    received = CORE[:60] + TAG + CORE[60:]
    construct, entries = _one_construct(tmp_path, received)
    codes = workflows.encode(received, "dayhoff6")
    frequencies = workflows._letter_frequencies(codes, "dayhoff6")
    short = workflows.region_evalue(codes[60:70], frequencies, len(received), len(TAG))
    long = workflows.region_evalue(codes[60:95], frequencies, len(received), len(TAG))
    assert long < short


def test_diagonal_runs_report_one_maximal_run(tmp_path):
    received = CORE[:30] + TAG + CORE[30:]
    received_codes = workflows.encode(received, "dayhoff6")
    insert_codes = workflows.encode(TAG, "dayhoff6")
    runs = workflows._diagonal_runs(received_codes, insert_codes, 10)
    best = max(runs, key=lambda r: r[2])
    assert best[0] == 30 and best[2] == len(TAG)


# --- evaluation -------------------------------------------------------------------


def test_overlap_is_intersection_over_union():
    assert evaluate.overlap((1, 10), (1, 10)) == 1.0
    assert evaluate.overlap((1, 10), (11, 20)) == 0.0
    assert evaluate.overlap((1, 10), (6, 15)) == pytest.approx(5 / 15)


def test_event_matching_needs_the_same_name_and_enough_overlap():
    event = records.Event(
        "C1", "v1", "E1", "SUMO tail", TAG, 61, 60 + len(TAG),
        "unexpected_insert", "sanger", date(2024, 5, 1),
    )
    flag = workflows.Flag(
        "reduced", "C1", "v1", 61, 60 + len(TAG), TAG, "SUMO tail", "", False,
        "unexpected", None, "dayhoff6, k=10",
    )
    assert evaluate.matches(flag, event)
    assert not evaluate.matches(
        workflows.Flag(
            "reduced", "C1", "v1", 61, 60 + len(TAG), TAG, "His6", "", False,
            "unexpected", None, "dayhoff6, k=10",
        ),
        event,
    )
    assert not evaluate.matches(
        workflows.Flag(
            "reduced", "C1", "v1", 200, 210, TAG, "SUMO tail", "", False,
            "unexpected", None, "dayhoff6, k=10",
        ),
        event,
    )


def test_an_ambiguous_flag_is_not_a_recovery_but_is_a_false_flag(tmp_path):
    rows = [
        construct_row(construct_id="C1", group_id="G1", status="error",
                      received_protein_seq=CORE[:60] + TAG + CORE[60:]),
        construct_row(construct_id="C2", group_id="G2", status="clean"),
    ]
    events = [
        {
            "construct_id": "C1",
            "version": "v1",
            "event_id": "E1",
            "insert_name": "SUMO tail",
            "insert_seq": TAG,
            "start": 61,
            "end": 60 + len(TAG),
            "decision": "unexpected_insert",
            "evidence": "sanger trace",
            "decided_date": "2024-05-01",
        }
    ]
    directory = write_collection(tmp_path, rows, events=events, dictionary_rows=[dictionary_row()])
    collection = records.load(directory)
    flags = [
        workflows.Flag("reduced", "C1", "v1", 61, 60 + len(TAG), TAG, "SUMO tail",
                       "", False, "ambiguous", None, "dayhoff6, k=10"),
        workflows.Flag("reduced", "C2", "v1", 5, 20, CORE[4:20], "SUMO tail",
                       "", False, "ambiguous", None, "dayhoff6, k=10"),
    ]
    scores = evaluate.score(collection, {"reduced": flags})["reduced"]
    assert scores.events_total == 1 and scores.events_recovered == 0
    assert scores.clean_constructs == 1 and scores.clean_constructs_flagged == 1


def test_a_permitted_flag_is_not_a_false_flag(tmp_path):
    directory = write_collection(
        tmp_path, [construct_row(status="clean")], dictionary_rows=[dictionary_row()]
    )
    collection = records.load(directory)
    flags = [
        workflows.Flag("reduced", "C1", "v1", 1, len(TAG), TAG, "SUMO tail",
                       "", True, "permitted", None, "dayhoff6, k=10")
    ]
    scores = evaluate.score(collection, {"reduced": flags})["reduced"]
    assert scores.clean_constructs_flagged == 0


def test_insert_classes_split_short_repetitive_and_hydrophobic():
    assert "short" in evaluate.insert_classes("HHHHHH")
    assert "repetitive" in evaluate.insert_classes("GGGGSGGGGSGGGGS")
    assert "hydrophobic" in evaluate.insert_classes("AILVFWMAILVFWMA")
    assert evaluate.insert_classes(TAG) == ["ordinary"]


def test_pass_rule_without_review_sessions_is_insufficient_evidence(tmp_path):
    directory = write_collection(
        tmp_path, [construct_row(status="clean")], dictionary_rows=[dictionary_row()]
    )
    collection = records.load(directory)
    scores = evaluate.score(collection, {"reduced": [], "comparison": []})
    rule = evaluate.apply_pass_rule(collection, scores)
    assert rule.verdict == "INSUFFICIENT_EVIDENCE"
    assert rule.review_time_ok is None
    assert any("machine runtime is not review time" in r for r in rule.reasons)


def test_enough_data_names_every_shortfall(tmp_path):
    directory = write_collection(
        tmp_path, [construct_row()], dictionary_rows=[dictionary_row()]
    )
    collection = records.load(directory)
    enough, shortfalls = evaluate.enough_data(collection)
    assert not enough
    assert len(shortfalls) == 4


# --- controls ---------------------------------------------------------------------


def test_neighbour_preserving_shuffle_keeps_composition():
    rng = random.Random(0)
    for seq in (TAG, "GGGGSGGGGSGGGGS", CORE[:40]):
        shuffled = controls.neighbour_preserving_shuffle(seq, rng)
        assert sorted(shuffled) == sorted(seq)


def test_uniform_shuffle_keeps_composition():
    rng = random.Random(0)
    shuffled = controls.uniform_shuffle(TAG, rng)
    assert sorted(shuffled) == sorted(TAG)


def test_composition_matched_stretch_has_the_insert_length():
    rng = random.Random(0)
    stretch = controls.composition_matched_stretch(TAG, CORE, rng)
    assert len(stretch) == len(TAG)
    assert stretch in CORE
