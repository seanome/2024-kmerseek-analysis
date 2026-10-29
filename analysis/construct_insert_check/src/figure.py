"""Draw the paired table and the flag rates from a run's output.

    python src/figure.py --results <output dir> --figure figures/<stem>

Both panels are drawn from ``paired_events.csv`` and ``summary.json`` written by
``cli.py run``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "notebooks"))

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pubfig as pf

#: One colour per workflow, kept the same in both panels.
WORKFLOW_COLOUR = {
    "exact": pf.OKABE_ITO["orange"],
    "align": pf.OKABE_ITO["bluish_green"],
    "dna": pf.OKABE_ITO["reddish_purple"],
    "comparison": pf.OKABE_ITO["black"],
    "reduced": pf.OKABE_ITO["blue"],
}
WORKFLOW_LABEL = {
    "exact": "exact tag search\n(all 20 amino acids)",
    "align": "intended against\nreceived alignment",
    "dna": "DNA translation",
    "comparison": "the three together",
    "reduced": "reduced alphabet\nk-mer search",
}
ORDER = ["exact", "align", "dna", "comparison", "reduced"]

CAUGHT = pf.OKABE_ITO["blue"]
MISSED = pf.GREY


def draw(results: Path, stem: Path, source: str) -> None:
    pf.use_style()
    events = list(csv.DictReader((results / "paired_events.csv").open()))
    summary = json.loads((results / "summary.json").read_text())
    scores = summary["scores"]

    events.sort(key=lambda r: (int(r["insert_length"]), r["event_id"]))
    lengths = [int(r["insert_length"]) for r in events]

    fig, (ax_grid, ax_rate) = pf.figure(
        pf.TWO_COLUMN_MM, 78, ncols=2, width_ratios=[2.2, 1]
    )

    x = np.arange(len(events))
    for row, workflow in enumerate(ORDER):
        caught = np.array([r.get(workflow) == "caught" for r in events])
        y = np.full(len(events), len(ORDER) - 1 - row, dtype=float)
        ax_grid.scatter(
            x[caught],
            y[caught],
            marker="s",
            s=10,
            color=CAUGHT,
            label="panel a: caught, counted as a recovery" if row == 0 else None,
        )
        ax_grid.scatter(
            x[~caught],
            y[~caught],
            marker="o",
            s=8,
            facecolors="none",
            edgecolors=MISSED,
            linewidths=0.5,
            label="panel a: missed" if row == 0 else None,
        )
    ax_grid.set_yticks(range(len(ORDER)))
    ax_grid.set_yticklabels([WORKFLOW_LABEL[w] for w in reversed(ORDER)])
    for y in range(len(ORDER)):
        ax_grid.axhline(y, color="#DDDDDD", linewidth=0.3, zorder=0)
    ax_grid.set_xlabel("Each documented unexpected insert, ordered by its length (aa)")
    ax_grid.set_ylim(-0.6, len(ORDER) - 0.4)
    step = max(1, len(events) // 8)
    ticks = list(range(0, len(events), step))
    ax_grid.set_xticks(ticks)
    ax_grid.set_xticklabels([lengths[t] for t in ticks])

    rows = []
    for workflow in ORDER:
        score = scores[workflow]
        recall = score["event_recall"]
        clean = score["clean_flag_rate"]
        # None means not assessed. It is left out, never drawn as 0%.
        rows.append(
            (
                workflow,
                np.nan if recall is None else 100 * recall,
                np.nan if clean is None else 100 * clean,
            )
        )
    y = np.arange(len(rows))[::-1]
    ax_rate.scatter(
        [r[1] for r in rows],
        y,
        marker="o",
        s=14,
        color=[WORKFLOW_COLOUR[r[0]] for r in rows],
    )
    ax_rate.scatter(
        [r[2] for r in rows],
        y,
        marker="s",
        s=14,
        facecolors="none",
        edgecolors=[WORKFLOW_COLOUR[r[0]] for r in rows],
        linewidths=0.8,
    )
    for position, row in zip(y, rows):
        ax_rate.axhline(position, color="#DDDDDD", linewidth=0.3, zorder=0)
    ax_rate.axvline(
        95, color=pf.OKABE_ITO["vermillion"], linewidth=0.5, linestyle=(0, (3, 2))
    )
    ax_rate.axvline(
        1, color=pf.OKABE_ITO["vermillion"], linewidth=0.5, linestyle=(0, (1, 2))
    )
    ax_rate.text(
        93,
        len(rows) - 0.42,
        "95% of inserts\nmust be recovered",
        fontsize=5,
        color=pf.OKABE_ITO["vermillion"],
        ha="right",
        va="top",
    )
    ax_rate.text(
        35,
        0.5,
        "at most 1% of clean\nconstructs may be flagged",
        fontsize=5,
        color=pf.OKABE_ITO["vermillion"],
        ha="left",
        va="center",
    )
    ax_rate.set_yticks(y)
    ax_rate.set_yticklabels([WORKFLOW_LABEL[r[0]] for r in rows])
    ax_rate.set_xlim(-3, 103)
    ax_rate.set_xticks([0, 25, 50, 75, 100])
    ax_rate.set_xlabel("Per cent (higher recovery is better,\nfewer false flags is better)")

    fig.get_layout_engine().set(rect=(0, 0, 1, 0.80))
    proxies = [
        Line2D([], [], marker="o", linestyle="none", markersize=3.5,
               markerfacecolor="#444444", markeredgecolor="#444444",
               label="panel b: inserts recovered, of all documented (%)"),
        Line2D([], [], marker="s", linestyle="none", markersize=3.5,
               markerfacecolor="none", markeredgecolor="#444444",
               label="panel b: clean constructs flagged (%)"),
    ]
    handles, labels = {}, []
    for proxy in proxies:
        handles[proxy.get_label()] = proxy
        labels.append(proxy.get_label())
    for ax in (ax_grid, ax_rate):
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in handles:
                handles[label] = handle
                labels.append(label)
    fig.legend(
        [handles[label] for label in labels],
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.86),
        ncol=2,
        frameon=False,
    )
    for ax, letter in zip((ax_grid, ax_rate), "ab"):
        pf.panel_label(ax, letter)

    counts = summary["counts"]
    header = (
        f"Workflows compared on {source}: {counts['construct_versions']} construct "
        f"versions, {counts['documented_events']} documented unexpected inserts, "
        f"{counts['clean_constructs']} clean constructs.\n"
        f"Verdict {summary['verdict']}. The reduced-alphabet search recovered "
        f"{len(summary['pass_rule']['events_the_comparison_missed'])} inserts the three "
        "other workflows missed; five are required."
    )
    fig.text(0.01, 0.99, header, fontsize=6, ha="left", va="top")
    pf.save(fig, stem)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True)
    parser.add_argument("--figure", required=True)
    parser.add_argument(
        "--source",
        default="synthetic records, mechanics only",
        help="what the records are, written on the figure",
    )
    args = parser.parse_args()
    draw(Path(args.results), Path(args.figure), args.source)
    print(f"wrote {args.figure}.pdf, .svg and .png")
    plt.close("all")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
