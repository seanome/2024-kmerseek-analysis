"""Plot the current review ledger; counts apply only to the recorded snapshot."""

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks"))
import pubfig as pf
from render_notebook_review_status import review_state


def main():
    directory = Path(sys.argv[1])
    snapshot = json.loads((directory / "open_prs.json").read_text())
    reviews = {
        r["number"]: r for r in json.loads((directory / "reviews.json").read_text())
    }
    counts = Counter(
        review_state(p, reviews.get(p["number"])) for p in snapshot["pull_requests"]
    )
    labels = [
        "Not reviewed",
        "Head changed; review needs updating",
        "Partial review",
        "Reviewed; blockers remain",
        "Reviewed; no recorded blockers",
    ]
    pf.use_style()
    fig, ax = pf.figure(pf.TWO_COLUMN_MM, 70, layout=None)
    fig.subplots_adjust(left=0.46, right=0.94, bottom=0.20, top=0.78)
    ax.scatter(
        [counts[x] for x in labels],
        range(len(labels)),
        color=pf.OKABE_ITO["blue"],
        label="PR count at the recorded snapshot",
    )
    ax.set_yticks(range(len(labels)), labels)
    ax.invert_yaxis()
    ax.set_xlim(-1, max(counts.values()) + 6)
    ax.set_xlabel("Pull requests (count)")
    ax.grid(axis="y")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01))
    for i, label in enumerate(labels):
        ax.annotate(
            str(counts[label]),
            (counts[label], i),
            xytext=(5, 0),
            textcoords="offset points",
            va="center",
        )
    fig.text(
        0.03,
        0.97,
        "Open analysis PR review: incomplete reviews remain visible",
        va="top",
    )
    fig.text(
        0.03,
        0.03,
        "Source: open_prs.json and reviews.json. A clean CI result is not scientific review.",
    )
    pf.save(fig, directory / "review_coverage")


if __name__ == "__main__":
    main()
