import json, re, sys, datetime

d = json.load(open(sys.argv[1]))
WHAT = json.load(open(sys.argv[2])) if len(sys.argv) > 2 else {}
HUMAN = json.load(open("data/2026-W43_human_reviews.json"))
ORDER = ["seanome/kmerseek", "seanome/2024-kmerseek-analysis"]


def md(s):
    return datetime.date.fromisoformat(s).strftime("%b %-d")


def short(t):
    t = re.sub(r"\s*\([^)]*\)?$", "", t)
    w = t.split()
    return " ".join(w[:10]) + (" …" if len(w) > 10 else "")


def what(repo, p):
    return WHAT.get(f"{repo}#{p['number']}") or short(p["title"])


def status(p):
    if p["merged"]:
        return f"Merged {md(p['merged'])}"
    if p["state"] == "closed":
        return f"Closed, not merged {md(p['closed'])}"
    return "Open (draft)" if p["draft"] else "Open"


def review_mark(p):
    c = p["skill_review"]["pr_comments"]
    if not c:
        return "☐"
    last = max(c, key=lambda x: x["date"])
    if last["is_head"] is True:
        return f"✅ {md(last['date'])}"
    return f"⚠️ {md(last['date'])}, commits since"


def reasons(repo, p):
    if p["state"] != "open":
        return []
    r = []
    ev = HUMAN[f"{repo}#{p['number']}"]
    if not ev:
        r.append("**Not reviewed.**")
    elif p["last_commit_at"] and p["last_commit_at"] > ev[-1][0]:
        r.append("**Your comments, then new commits; needs another look.**")
    else:
        r.append(f"Your comments of {md(ev[-1][0][:10])}; no commits since.")
    b = p["base_status"]
    k = b["kind"]
    if k == "stacked_on_open_pr":
        r.append(f"Targets #{b['prs'][0]['number']}.")
    elif k == "base_already_in_main":
        s = "Its target branch is already in `main`; retarget to `main`."
        if b["ahead_of_main"]:
            s = (f"Its target `{b['branch']}` was merged, but has {b['ahead_of_main']} "
                 "later commits not in `main`.")
        r.append(s)
    elif k == "base_pr_closed_unmerged":
        n = b["prs"][0]["number"]
        r.append(f"Its target `{b['branch']}` had PR #{n} closed without merging, "
                 "so there is no route into `main`.")
    elif k == "base_has_no_pr":
        r.append(f"Its target `{b['branch']}` ({b['ahead_of_main']} commits ahead of `main`) has no PR.")
    if p["mergeable_state"] == "dirty":
        r.append("Merge conflicts.")
    if p["checks_failed"]:
        r.append("Failing: " + ", ".join(p["checks_failed"]) + ".")
    if repo == "seanome/kmerseek" and p["checks_ran"] == ["Check Rust Formatting"]:
        r.append("Only the formatting check ran, not the tests.")
    return r


def link(repo, n):
    return f"[#{n}](https://github.com/{repo}/pull/{n})"


def linkify(repo, s):
    return re.sub(r"#(\d+)", lambda m: link(repo, m.group(1)), s)


out = []
todo = []
for repo in ORDER:
    r = d["repos"][repo]
    prs = r["prs"]
    skill = prs[0]["skill_review"]["skill"]
    out.append(f"\n### {repo}\n")
    out.append(f"| PR | What it does | Status | {skill} | Why it's still open |")
    out.append("|---|---|---|---|---|")
    rows = ([p for p in prs if p["merged"]] + sorted([p for p in prs if p["state"] == "open"], key=lambda p: p["number"])
            + [p for p in prs if p["state"] == "closed" and not p["merged"]])
    for p in rows:
        rs = linkify(repo, " ".join(reasons(repo, p)))
        out.append(f"| {link(repo, p['number'])} | {what(repo, p)} | {status(p)} | {review_mark(p)} | {rs} |")

    # review order: stacks first (roots with most descendants), depth-first; then the rest
    op = {p["number"]: p for p in prs if p["state"] == "open"}
    parent = {n: p["base_status"]["prs"][0]["number"] for n, p in op.items()
              if p["base_status"]["kind"] == "stacked_on_open_pr"}
    kids = {}
    for c, pa in parent.items():
        kids.setdefault(pa, []).append(c)

    def size(n):
        return 1 + sum(size(c) for c in kids.get(n, []))

    def dfs(n, depth):
        yield n, depth
        for c in sorted(kids.get(n, [])):
            yield from dfs(c, depth + 1)

    roots = [n for n in op if n not in parent]
    stacked = sorted([n for n in roots if n in kids], key=lambda n: (-size(n), n))
    rest = [n for n in roots if n not in kids]

    def urgency(n):
        p = op[n]
        needs_look = bool(HUMAN[f"{repo}#{n}"])
        return (0 if needs_look else 1, -(int(p["last_commit"].replace("-", "")) if p["last_commit"] else 0))

    rest.sort(key=urgency)
    todo.append(f"\n## {repo}\n")
    for root in stacked:
        for n, depth in dfs(root, 0):
            p = op[n]
            rs = " ".join(x.replace("**", "") for x in reasons(repo, p))
            todo.append(f"- [ ] {link(repo, n)} {what(repo, p)}: {linkify(repo, rs)}")
    for n in rest:
        p = op[n]
        rs = " ".join(x.replace("**", "") for x in reasons(repo, p))
        todo.append(f"- [ ] {link(repo, n)} {what(repo, p)}: {linkify(repo, rs)}")
    listed = sum(len(list(dfs(root, 0))) for root in stacked) + len(rest)
    assert listed == len(op), f"{repo}: {listed} to-do entries for {len(op)} open PRs"

open("tables.md", "w").write("\n".join(out) + "\n")
open("todo.md", "w").write("\n".join(todo) + "\n")
