import json
import subprocess
from pathlib import Path

HOOK = (
    Path(__file__).resolve().parents[2]
    / ".claude"
    / "hooks"
    / "require_review_check.py"
)

NOTE = "> [!NOTE]\n> Written by Claude Code at olgabot's request.\n\n"
HEAD = NOTE + "## /analysis-review, run 2026-10-01 12:00 UTC\n\n## Verdict\nFine.\n\n"
CHECKED = (
    "## Preparing for an Adversarial Reviewer 2: Stress-testing the findings\n"
    "- Headline re-computed: 213 by a second polars query: 213 vs 213.\n"
    "- Not tested: the Sherlock inputs, not reachable from the Mac.\n\n"
)
MARKER = "<!-- analysis-review sha=abc date=2026-10-01 -->\n"
POST = (
    "gh api -X POST repos/seanome/2024-kmerseek-analysis/issues/56/comments -F body=@{}"
)


def run_hook(command, cwd):
    payload = {"tool_name": "Bash", "tool_input": {"command": command}, "cwd": str(cwd)}
    out = subprocess.run(
        ["/usr/bin/python3", str(HOOK)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return (
        json.loads(out)["hookSpecificOutput"]["permissionDecisionReason"]
        if out
        else None
    )


def post(tmp_path, body):
    (tmp_path / "review.md").write_text(body)
    return run_hook(POST.format("review.md"), tmp_path)


def test_review_with_the_section_is_posted(tmp_path):
    assert post(tmp_path, HEAD + CHECKED + MARKER) is None


def test_review_without_the_section_is_refused(tmp_path):
    assert (
        'no "## Preparing for an Adversarial Reviewer 2: Stress-testing the findings" section'
        in post(tmp_path, HEAD + MARKER)
    )


def test_section_with_no_lines_is_refused(tmp_path):
    body = (
        HEAD
        + "## Preparing for an Adversarial Reviewer 2: Stress-testing the findings\n\n"
        + MARKER
    )
    assert 'no "- " lines' in post(tmp_path, body)


def test_section_without_not_tested_is_refused(tmp_path):
    body = (
        HEAD
        + "## Preparing for an Adversarial Reviewer 2: Stress-testing the findings\n- Re-ran cell 7.\n\n"
        + MARKER
    )
    assert 'no "Not tested" line' in post(tmp_path, body)


def test_not_tested_in_a_later_section_does_not_count(tmp_path):
    body = (
        HEAD
        + "## Preparing for an Adversarial Reviewer 2: Stress-testing the findings\n- Re-ran cell 7.\n\n## Other\nNot tested: x\n"
    )
    assert 'no "Not tested" line' in post(tmp_path, body)


def test_other_comments_are_not_checked(tmp_path):
    assert (
        post(tmp_path, NOTE + "## /rustacean-review, run today\nLooks fine.\n") is None
    )


def test_reading_a_comment_back_is_not_checked(tmp_path):
    cmd = "gh api repos/seanome/2024-kmerseek-analysis/issues/comments/1 --jq .body"
    assert run_hook(cmd, tmp_path) is None


def test_inline_body_is_checked(tmp_path):
    cmd = "gh api -X POST repos/o/r/issues/1/comments -f body='## /analysis-review, run x'"
    assert "section" in run_hook(cmd, tmp_path)


def test_body_piped_in_is_checked(tmp_path):
    (tmp_path / "review.md").write_text(HEAD + MARKER)
    stdin_post = POST.format("-")
    for cmd in (f"cat review.md | {stdin_post}", f"{stdin_post} < review.md"):
        assert "Reviewer 2" in (run_hook(cmd, tmp_path) or ""), cmd
