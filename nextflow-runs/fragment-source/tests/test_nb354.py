"""Tests for the scoring functions in bin/nb354.py that set the verdict numbers."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bin"))

import lookup  # noqa: E402
import nb354  # noqa: E402


def _scores(tops):
    """One shuffled call per top score; each has a single candidate."""
    return {f"s{i}": [(t, "X")] for i, t in enumerate(tops)}


def test_threshold_one_percent_of_100_shuffles_may_reach_it():
    # 100 shuffles, so one may reach the threshold. The two highest top scores are 100 and 50.
    tops = [100, 50] + [10] * 98
    scores = _scores(tops)
    assert nb354.threshold_from_shuffles(scores, list(scores)) == 51


def test_threshold_ties_at_the_second_highest_score_are_excluded():
    # Two shuffles tie at 80. Only one may reach t, so t must be 81 and only the 90 reaches it.
    tops = [90, 80, 80] + [5] * 97
    scores = _scores(tops)
    t = nb354.threshold_from_shuffles(scores, list(scores))
    assert t == 81
    assert sum(x >= t for x in tops) == 1


def test_threshold_with_fewer_than_100_shuffles_beats_every_shuffle():
    # 50 shuffles: int(0.01 * 50) = 0 may reach t, so t is one above the highest.
    scores = _scores([30, 20] + [1] * 48)
    assert nb354.threshold_from_shuffles(scores, list(scores)) == 31


def test_threshold_shuffles_with_no_candidates_score_zero():
    scores = {"a": []}
    assert nb354.threshold_from_shuffles(scores, ["a", "missing"]) == 1
    assert nb354.threshold_from_shuffles({}, []) == 1


def test_recovered_takes_top_five_passing_and_threshold_is_inclusive():
    # Candidates arrive sorted by score then accession. At tau = 10 all six pass.
    ranked = [(12, "A"), (10, "B"), (10, "C"), (10, "D"), (10, "E"), (10, "F")]
    assert nb354.recovered(ranked, {"E"}, 10)  # rank 5 of 6
    assert not nb354.recovered(ranked, {"F"}, 10)  # rank 6, outside the top five
    assert not nb354.recovered(ranked, {"E"}, 11)  # E scores 10, below 11
    assert nb354.recovered(ranked, {"A"}, 11)


def test_recovered_with_no_candidates_is_false():
    assert not nb354.recovered([], {"A"}, 1)
    assert nb354.list_length([], 1) == 0
    assert nb354.top_score([]) == 0


def test_list_length_counts_scores_at_or_above_tau():
    ranked = [(9, "A"), (7, "B"), (7, "C"), (3, "D")]
    assert nb354.list_length(ranked, 7) == 3
    assert nb354.list_length(ranked, 8) == 1
    assert nb354.list_length(ranked, 10) == 0


def test_recovery_curve_steps_at_each_score_and_one_above():
    present = [{"call_id": "p1", "accepted": {"A"}}, {"call_id": "p2", "accepted": {"B"}},
               {"call_id": "p3", "accepted": {"C"}}]
    sp = {"p1": [(20, "A")], "p2": [(10, "B")]}  # p3 has no candidates
    absent = [{"call_id": "a1"}, {"call_id": "a2"}]
    sa = {"a1": [(10, "Z")]}  # a2 has no candidates
    curve = {r["tau"]: r for r in nb354.recovery_curve(sp, sa, present, absent)}
    assert sorted(curve) == [1, 10, 11, 20, 21]
    assert curve[10]["recovery"] == 2 / 3 and curve[10]["false_assignment"] == 1 / 2
    assert curve[11]["recovery"] == 1 / 3 and curve[11]["false_assignment"] == 0
    assert curve[21]["recovery"] == 0


def test_blast_candidates_parses_accession_forms(tmp_path):
    out = tmp_path / "blast.tsv"
    out.write_text(
        "q1\tP12345\t90.0\n"
        "q1\tlcl|Q99999\t80.0\n"
        "q1\tsp|O11111|NAME_HUMAN\t70.0\n"
        "q1\ttr|A0A000|NAME\t60.0\n"
        "q1\tP12345\t50.0\n"  # repeated hit collapses
        "q2\tP22222\t40.0\n"
    )
    cands = lookup.blast_candidates(str(out))
    assert cands["q1"] == {"P12345", "Q99999", "O11111", "A0A000"}
    assert cands["q2"] == {"P22222"}
    assert "q3" not in cands
