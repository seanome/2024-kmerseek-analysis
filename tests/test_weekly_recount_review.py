"""Review provenance and pagination must not alter the human-review denominator."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "reports/weekly-pr-status/2026-W42_recount_human_reviews.py"
)
spec = importlib.util.spec_from_file_location("weekly_recount_review", SCRIPT)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_codex_and_claude_are_not_counted_as_human(monkeypatch):
    comments = [
        {
            "user": {"login": "olgabot"},
            "body": body,
            "created_at": "2026-10-18T12:00:00Z",
        }
        for body in [
            "Written by Codex (GPT-6).",
            "Written by Claude Code.",
            "Please check the denominator.",
        ]
    ]
    monkeypatch.setattr(m, "gh", lambda path: comments if "/issues/" in path else [])
    _, got = m.facts(
        "owner/repo", {"number": 1, "last_commit_at": "2026-10-18T13:00:00Z"}
    )
    assert got["n_human"] == 1
    assert got["commits_after"]


def test_whitespace_separated_pages_and_pager(monkeypatch):
    def run(args, **kw):
        assert kw["check"] is True
        assert kw["env"]["GH_PAGER"] == "cat"
        return SimpleNamespace(stdout='[{"id":1}]\n\n[{"id":2}]\n')

    monkeypatch.setenv("GH_PAGER", "less")
    monkeypatch.setattr(m.subprocess, "run", run)
    assert m.gh("x") == [{"id": 1}, {"id": 2}]


def test_nonlist_api_output_is_not_empty_review_evidence(monkeypatch):
    monkeypatch.setattr(
        m.subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(stdout='{"message":"failure"}'),
    )
    with pytest.raises(ValueError, match="GitHub list"):
        m.gh("x")
