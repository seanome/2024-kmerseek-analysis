"""Sparse detections retain all finite hits when below the false-positive budget."""

import sys
from pathlib import Path
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks"))
import e3_detection_gap_utils as e3


def test_no_negative_detections_do_not_suppress_positive_hits():
    p = pl.DataFrame({"label": [False, False, True], "score_phmmer": [None, None, 5.0]})
    called, thr = e3.add_calls(p, ["phmmer"], 0.01)
    assert called["call_phmmer"].to_list() == [False, False, True]


def test_within_budget_includes_lowest_score_and_excludes_nonfinite():
    p = pl.DataFrame(
        {
            "label": [False] * 4 + [True] * 3,
            "score_phmmer": [2.0, None, None, None, 2.0, float("nan"), float("inf")],
        }
    )
    called, thr = e3.add_calls(p, ["phmmer"], 0.25)
    assert called["call_phmmer"].to_list() == [
        True,
        False,
        False,
        False,
        True,
        False,
        False,
    ]
    assert thr["realised_fpr"][0] == 0.25


def test_zero_budget_ties_stay_below_budget():
    p = pl.DataFrame(
        {"label": [False, False, True, True], "score_phmmer": [3.0, 2.0, 3.0, 4.0]}
    )
    called, thr = e3.add_calls(p, ["phmmer"], 0.0)
    assert called["call_phmmer"].to_list() == [False, False, False, True]
    with pytest.raises(ValueError, match="no negative"):
        e3.threshold_at_fpr(p.filter("label"), "phmmer", 0.01)
