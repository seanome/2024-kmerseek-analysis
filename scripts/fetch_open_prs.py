"""Write data/open_prs.json: the open pull requests in the analysis and engine repos.

Uses the GitHub REST API. A GITHUB_TOKEN in the environment raises the rate
limit; both repos are public, so it also works without one. Run from the repo
root:

    python scripts/fetch_open_prs.py
"""

import datetime
import json
import os
import urllib.request
from pathlib import Path

REPOS = ["seanome/2024-kmerseek-analysis", "seanome/kmerseek"]
OUT = Path(__file__).resolve().parent.parent / "data" / "open_prs.json"


def get(url):
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json"})
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)


def main():
    prs = []
    for repo in REPOS:
        page = 1
        while True:
            batch = get(f"https://api.github.com/repos/{repo}/pulls?state=open&per_page=100&page={page}")
            prs += [{"repo": repo, "n": p["number"], "title": p["title"], "draft": p["draft"],
                     "branch": p["head"]["ref"], "updated": p["updated_at"][:10]} for p in batch]
            if len(batch) < 100:
                break
            page += 1
    fetched = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    OUT.write_text(json.dumps({"fetched": fetched, "prs": prs}, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {OUT.name}: {len(prs)} open pull requests, fetched {fetched}")


if __name__ == "__main__":
    main()
