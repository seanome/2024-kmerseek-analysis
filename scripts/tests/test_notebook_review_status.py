"""Prevent missing or stale reviews from being reported as complete."""

import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "review_status", Path(__file__).parents[1] / "render_notebook_review_status.py"
)
review_status = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(review_status)


def snapshot():
    return {
        "captured_at": "2026-10-23T12:00:00Z",
        "pull_requests": [
            {
                "number": 1,
                "title": "A notebook",
                "head": "new-head",
                "github_state": "CLEAN",
                "url": "https://github.com/example/repo/pull/1",
            }
        ],
    }


def test_new_head_cannot_inherit_a_clean_review():
    pr = snapshot()["pull_requests"][0]
    review = {"number": 1, "head": "old-head", "complete": True, "blockers": []}
    assert (
        review_status.review_state(pr, review) == "Head changed; review needs updating"
    )


def test_partial_review_and_unreviewed_pr_stay_visible():
    pr = snapshot()["pull_requests"][0]
    review = {"number": 1, "head": "new-head", "complete": False, "blockers": []}
    assert review_status.review_state(pr, review) == "Partial review"
    text = review_status.render(snapshot(), [])
    assert "Not reviewed | 1" in text
    assert "Full review pending" in text
    assert "PR 1" in text


def test_completed_review_with_blockers_is_not_clean():
    pr = snapshot()["pull_requests"][0]
    review = {
        "number": 1,
        "head": "new-head",
        "complete": True,
        "blockers": ["Missing input data"],
    }
    assert review_status.review_state(pr, review) == "Reviewed; blockers remain"


def test_duplicate_or_out_of_scope_records_are_rejected():
    review = {"number": 1, "head": "new-head", "complete": False}
    with pytest.raises(ValueError, match="Duplicate review"):
        review_status.render(snapshot(), [review, review])
    with pytest.raises(ValueError, match="absent from snapshot"):
        review_status.render(snapshot(), [{**review, "number": 2}])
