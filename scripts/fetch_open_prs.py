"""Write data/open_prs.json: the open pull requests in the analysis and engine repos.

For each open pull request it records the branch it merges into and, in the
analysis repo, the notebooks and figure files the pull request itself adds or
changes. A pull request built on another one (its base branch is that one's
branch) lists only the files that are not already in the one below it, so a
notebook is listed on the pull request that brings it in.

It also records the state (merged, closed) of every pull request that
data/paper_map.json names but that is no longer open.

Uses the GitHub REST API. A GITHUB_TOKEN (or GH_TOKEN) in the environment
raises the rate limit from 60 to 5000 requests an hour; this script makes
about 100. Run from the repo root:

    GH_TOKEN=$(gh auth token) python scripts/fetch_open_prs.py
"""

import datetime
import json
import os
import re
import urllib.request
from pathlib import Path

REPOS = ["seanome/2024-kmerseek-analysis", "seanome/kmerseek"]
ANALYSIS = REPOS[0]
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "open_prs.json"
MAP = ROOT / "data" / "paper_map.json"
FIGURE_EXT = (".png", ".svg", ".pdf")


def get(url):
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json"})
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)


def paged(url):
    out, page = [], 1
    while True:
        batch = get(f"{url}{'&' if '?' in url else '?'}per_page=100&page={page}")
        out += batch
        if len(batch) < 100:
            return out
        page += 1


def notebook_id(path):
    """'notebooks/241_alphabet_ranking.ipynb' -> '241'; None for a notebook with no number."""
    m = re.match(r"(\d+[a-z]?)[_\-.]", path.rsplit("/", 1)[-1])
    return m.group(1) if m else None


def figures(paths):
    """Figure files, one per stem: the PNG when there is one, else the SVG, else the PDF."""
    by_stem = {}
    for p in paths:
        if p.endswith(FIGURE_EXT):
            stem = p.rsplit(".", 1)[0]
            by_stem.setdefault(stem, []).append(p)
    return sorted(
        min(v, key=lambda p: FIGURE_EXT.index(p[p.rindex(".") :]))
        for v in by_stem.values()
    )


def main():
    prs = []
    for repo in REPOS:
        for p in paged(f"https://api.github.com/repos/{repo}/pulls?state=open"):
            row = {
                "repo": repo,
                "n": p["number"],
                "title": p["title"],
                "draft": p["draft"],
                "branch": p["head"]["ref"],
                "base": p["base"]["ref"],
                "created": p["created_at"][:10],
                "updated": p["updated_at"][:10],
            }
            if repo == ANALYSIS:
                row["files"] = [
                    f["filename"]
                    for f in paged(
                        f"https://api.github.com/repos/{repo}/pulls/{p['number']}/files"
                    )
                ]
            prs.append(row)

    # A pull request built on another one keeps only the files the lower one does not have.
    head_of = {(p["repo"], p["branch"]): p for p in prs}
    for p in prs:
        if "files" not in p:
            continue
        below = head_of.get((p["repo"], p["base"]))
        p["base_pr"] = below["n"] if below else None
        own = [
            f for f in p["files"] if not below or f not in set(below.get("files", []))
        ]
        p["notebooks"] = sorted({f for f in own if f.endswith(".ipynb")})
        p["notebook_ids"] = sorted({i for i in map(notebook_id, p["notebooks"]) if i})
        p["figures"] = figures(own)
        p["n_files"] = len(p["files"])
    for p in prs:
        p.pop("files", None)

    # Pull requests the map names that are not open: merged or closed, and when.
    named = set()
    if MAP.exists():
        m = json.loads(MAP.read_text())
        for item in m.get("paper", []) + m.get("runs", []):
            named.update((item.get("repo", ANALYSIS), n) for n in item.get("prs", []))
        named.update((ANALYSIS, n) for n in m.get("pr_notes", {}).get(ANALYSIS, {}))
        named.update(
            ("seanome/kmerseek", int(n))
            for n in m.get("pr_notes", {}).get("seanome/kmerseek", {})
        )
    open_keys = {(p["repo"], p["n"]) for p in prs}
    closed = []
    for repo, n in sorted(named - open_keys):
        p = get(f"https://api.github.com/repos/{repo}/pulls/{n}")
        closed.append(
            {
                "repo": repo,
                "n": n,
                "title": p["title"],
                "branch": p["head"]["ref"],
                "base": p["base"]["ref"],
                "state": "merged" if p["merged_at"] else "closed",
                "date": (p["merged_at"] or p["closed_at"] or "")[:10],
            }
        )

    fetched = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%d %H:%M UTC"
    )
    OUT.write_text(
        json.dumps(
            {"fetched": fetched, "prs": prs, "not_open": closed},
            indent=1,
            ensure_ascii=False,
        )
        + "\n"
    )
    print(
        f"wrote {OUT.name}: {len(prs)} open pull requests and {len(closed)} named ones no longer open, fetched {fetched}"
    )


if __name__ == "__main__":
    main()
