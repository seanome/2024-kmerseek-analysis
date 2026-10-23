#!/usr/bin/env python3
"""Render the open-PR review ledger without treating stale reviews as current."""

import argparse
import json
from collections import Counter
from pathlib import Path


def review_state(pr, review):
    """Require a matching head before using a completed review's verdict."""
    if review is None:
        return "Not reviewed"
    if review["head"] != pr["head"]:
        return "Head changed; review needs updating"
    if not review.get("complete", False):
        return "Partial review"
    if review.get("blockers"):
        return "Reviewed; blockers remain"
    return "Reviewed; no recorded blockers"


def render(snapshot, reviews):
    """Build one row per snapshotted PR, retaining unreviewed and blocked rows."""
    prs = snapshot["pull_requests"]
    numbers = [p["number"] for p in prs]
    if len(numbers) != len(set(numbers)):
        raise ValueError("Duplicate PR numbers in the analysis snapshot")
    review_numbers = [r["number"] for r in reviews]
    if len(review_numbers) != len(set(review_numbers)):
        raise ValueError("Duplicate review records; keep the latest review explicitly")
    by_number = {r["number"]: r for r in reviews}
    unknown = set(by_number) - set(numbers)
    if unknown:
        raise ValueError(f"Reviews absent from snapshot: {sorted(unknown)}")
    states = Counter(review_state(p, by_number.get(p["number"])) for p in prs)
    lines = [
        "# Open notebook PR reviews, 2026-10-23",
        "",
        "```mermaid",
        "flowchart LR",
        "  A[Open PR heads] --> B[Read notebooks and helpers]",
        "  B --> C[Recompute claims and fix errors]",
        "  C --> D[Test, push, and record remaining blockers]",
        "```",
        "",
        "This is an ongoing review of every open analysis PR. A mechanical check",
        "does not establish that a scientific claim is correct. A review only applies",
        "to its recorded commit; a newer head needs another check.",
        "",
        f"Snapshot taken: {snapshot['captured_at']}.",
        "",
        "| Review coverage | PRs |",
        "|---|---:|",
    ]
    lines.extend(f"| {state} | {count} |" for state, count in sorted(states.items()))
    lines.extend(
        [
            "",
            "![Review coverage at the recorded snapshot](review_coverage.png)",
            "",
            "## Each PR",
            "",
            "CI and conflicts are GitHub's state at the snapshot time. Neither is",
            "a substitute for reviewing the notebook's conclusions. An incomplete",
            "experiment can merge as documented preparation without establishing its result.",
            "",
            "| PR | Title | Head | GitHub state | Review | Remaining work |",
            "|---|---|---|---|---|---|",
        ]
    )
    for pr in sorted(prs, key=lambda p: p["number"], reverse=True):
        review = by_number.get(pr["number"])
        state = review_state(pr, review)
        if review and review.get("comment_url"):
            state = f"[{state}]({review['comment_url']})"
        remaining = (
            "; ".join(review.get("blockers", [])) if review else "Full review pending"
        )
        title = pr["title"].replace("|", "\\|").replace("\n", " ")
        remaining = remaining.replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| [PR {pr['number']}]({pr['url']}) | {title} | "
            f"`{pr['head'][:8]}` | {pr['github_state']} | {state} | {remaining} |"
        )
    lines.extend(
        [
            "",
            "## Scope and evidence",
            "",
            "The user selected every open PR first, rather than all historical notebooks",
            "on main. Independent reviewers use separate worktrees. Each review comment",
            "states its tests, independent recounts, fixes, and unavailable inputs.",
            "",
            "The initial notebook inventory deduplicates identical Git blobs across PRs.",
            "Formatting-only changes can repeat old notebook defects across many PRs;",
            "they are recorded separately from changes to the scientific analysis.",
            "",
            "Written by Codex at olgabot's request.",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    snapshot = json.loads((args.directory / "open_prs.json").read_text())
    reviews = json.loads((args.directory / "reviews.json").read_text())
    (args.directory / "README.md").write_text(render(snapshot, reviews))


if __name__ == "__main__":
    main()
