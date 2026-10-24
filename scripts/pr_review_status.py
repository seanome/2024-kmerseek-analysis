"""Write data/pr_review.json: each open pull request's review status and what stops it merging.

Reads the JSON that the weekly-pr-status skill's pr_status.py prints, and applies
the same rules as the weekly review tables (first used in the week 41 review,
2026-10-13):

- "not reviewed": no review and no review comment from Olga.
- "look again": Olga reviewed, then new commits came.
- "reviewed, no commits since": Olga reviewed and nothing changed after.

Blockers, in this order: the base branch is already in main (retarget), the
base pull request was closed without merging (no route into main), the pull
request it is built on, merge conflicts, failing checks. The review-skill mark
is "current" when a review comment names the head commit, "older commit" when
one names an earlier commit, and "none" otherwise.

    python ~/.claude/skills/weekly-pr-status/pr_status.py --week 2026-W42 > prs.json
    python scripts/pr_review_status.py prs.json
"""

import datetime
import json
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "data" / "pr_review.json"


def review(p):
    if not p["human_reviews"] and not p["human_comments"]:
        return "not reviewed"
    return (
        "look again" if p["commits_after_last_review"] else "reviewed, no commits since"
    )


def blockers(p, head_to_n):
    out, base = [], p.get("base_status") or {}
    kind = base.get("kind")
    if kind == "base_already_in_main":
        out.append("its base branch is already in main: retarget to main")
    elif kind == "base_pr_closed_unmerged":
        out.append(
            f"its base branch's PR was closed without merging, so it has no route into main"
        )
    elif kind == "stacked_on_open_pr" and p["base"] in head_to_n:
        out.append(f"built on PR {head_to_n[p['base']]}")
    if p.get("mergeable_state") == "dirty":
        out.append("merge conflicts")
    if p.get("checks_failed"):
        out.append("fails " + ", ".join(p["checks_failed"]))
    return out


def skill_mark(p):
    comments = (p.get("skill_review") or {}).get("pr_comments") or []
    if not comments:
        return {"mark": "none"}
    last = max(comments, key=lambda c: c["date"])
    return {
        "mark": "current" if last["is_head"] else "older commit",
        "date": last["date"],
    }


def main():
    src = json.loads(Path(sys.argv[1]).read_text())
    out = {
        "about": "Review status of each open pull request, from pr_status.py "
        f"(weekly-pr-status skill) for {src['since']} to {src['until']}, "
        "by the rules in scripts/pr_review_status.py.",
        "since": src["since"],
        "until": src["until"],
        # When pr_status.py wrote its file, not when this script ran.
        "read": datetime.datetime.fromtimestamp(
            Path(sys.argv[1]).stat().st_mtime, datetime.timezone.utc
        ).strftime("%Y-%m-%d %H:%M UTC"),
        "repos": {},
    }
    for repo, r in src["repos"].items():
        head_to_n = {p["head"]: p["number"] for p in r["prs"] if p["state"] == "open"}
        rows = {}
        for p in r["prs"]:
            if p["state"] != "open":
                continue
            reviews = [x["at"] for x in p["human_reviews"]]
            rows[str(p["number"])] = {
                "review": review(p),
                "last_review": (p.get("last_human_review_at") or "")[:10]
                or (max(reviews) if reviews else None),
                "blockers": blockers(p, head_to_n),
                "skill": (p.get("skill_review") or {}).get("skill"),
                "skill_review": skill_mark(p),
                "size": p.get("size"),
                "last_commit": p.get("last_commit"),
            }
        out["repos"][repo] = rows
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print(
        f"wrote {OUT.name}: "
        + ", ".join(f"{repo} {len(v)}" for repo, v in out["repos"].items())
    )


if __name__ == "__main__":
    main()
