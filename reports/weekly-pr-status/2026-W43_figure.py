"""Open PRs in week 2026-W43 by review state and by what else holds them up."""
import json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "notebooks"))
import pubfig as pf

d = json.loads((HERE / "data/2026-W43_prs.json").read_text())
human = json.loads((HERE / "data/2026-W43_human_reviews.json").read_text())
REPOS = {"seanome/kmerseek": ("kmerseek", pf.OKABE_ITO["blue"]),
         "seanome/2024-kmerseek-analysis": ("2024-kmerseek-analysis", pf.OKABE_ITO["vermillion"])}

review = ["Not reviewed by you", "Your comments, then new commits", "Your comments, no commits since"]
blockers = ["Built on another open PR", "Target branch merged, PR not retargeted",
            "Target PR closed, no route into main", "Merge conflicts", "A CI check fails"]
counts = {}
for repo, r in d["repos"].items():
    rv = dict.fromkeys(review, 0)
    bl = dict.fromkeys(blockers, 0)
    for p in r["prs"]:
        if p["state"] != "open":
            continue
        ev = human[f"{repo}#{p['number']}"]
        if not ev:
            rv[review[0]] += 1
        elif p["last_commit_at"] > ev[-1][0]:
            rv[review[1]] += 1
        else:
            rv[review[2]] += 1
        k = p["base_status"]["kind"]
        bl[blockers[0]] += k == "stacked_on_open_pr"
        bl[blockers[1]] += k == "base_already_in_main"
        bl[blockers[2]] += k == "base_pr_closed_unmerged"
        bl[blockers[3]] += p["mergeable_state"] == "dirty"
        bl[blockers[4]] += bool(p["checks_failed"])
    counts[repo] = (rv, bl)
    print(repo, rv, bl)

pf.use_style()
fig, axs = pf.figure(pf.TWO_COLUMN_MM, 55, ncols=2)
for ax, cats, i in ((axs[0], review, 0), (axs[1], blockers, 1)):
    h = 0.38
    for j, (repo, (name, colour)) in enumerate(REPOS.items()):
        vals = [counts[repo][i][c] for c in cats]
        ys = [y + (j - 0.5) * h for y in range(len(cats))]
        ax.barh(ys, vals, height=h, color=colour, label=name)
        for y, v in zip(ys, vals):
            ax.text(v + 1, y, str(v), va="center")
    ax.set_yticks(range(len(cats)), cats)
    ax.invert_yaxis()
    ax.set_xlabel("Open PRs (n)")
axs[0].set_xlim(0, 125)
axs[1].set_xlim(0, 50)
pf.shared_legend(fig)
for ax, letter in zip(axs, "ab"):
    pf.panel_label(ax, letter)
pf.save(fig, str(HERE / "2026-W43_open_prs_review_and_blockers"))
