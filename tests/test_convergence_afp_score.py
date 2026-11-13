"""Score ties must not depend on hit-file row order."""

import importlib.util
from pathlib import Path

import pytest

PATH = (
    Path(__file__).resolve().parents[1] / "nextflow-runs/convergence-afp/bin/score.py"
)
spec = importlib.util.spec_from_file_location("convergence_afp_score", PATH)
score = importlib.util.module_from_spec(spec)
spec.loader.exec_module(score)


def test_average_precision_ties_are_order_independent():
    truth = {"positive": "other"}
    a = score.score_one(["positive", "negative"], [1, 1], truth, "query", 1)
    b = score.score_one(["negative", "positive"], [1, 1], truth, "query", 1)
    assert a["auprc"] == b["auprc"] == pytest.approx(0.5)
    assert a["n_found"] == b["n_found"] == 0


def test_missing_positive_stays_in_denominator():
    got = score.score_one(
        ["positive", "negative"], [2, 1], {"positive": "other"}, "query", 2
    )
    assert got["auprc"] == pytest.approx(0.5)
    assert got["recall"] == pytest.approx(0.5)


def test_own_lineage_does_not_change_average_precision():
    got = score.score_one(
        ["own", "positive", "negative"],
        [3, 1, 1],
        {"own": "query", "positive": "other"},
        "query",
        1,
    )
    assert got["auprc"] == pytest.approx(0.5)


def test_empty_list_retrieves_nothing():
    got = score.score_one([], [], {}, "query", 1)
    assert got["auprc"] == 0
    assert got["n_found"] == 0


def test_negative_infinity_scores_are_one_tie():
    got = score.score_one(
        ["positive", "negative"],
        [-float("inf"), -float("inf")],
        {"positive": "other"},
        "query",
        1,
    )
    assert got["auprc"] == pytest.approx(0.5)
