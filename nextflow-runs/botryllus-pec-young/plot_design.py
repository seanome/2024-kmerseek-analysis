"""Figure: every young-colony library in the run, by time window, treatment and read depth.

Reads samples.csv (written by make_design.py) and writes figures/design_libraries_per_window.{pdf,svg,png}.
Run from this folder: python plot_design.py
"""

import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "notebooks"))
import numpy as np  # noqa: E402
import pubfig as pf  # noqa: E402

WINDOW_LABELS = {
    "young_2h": "2 h",
    "young_24h": "24 h",
    "young_13_17d": "13-17 d",
    "young_21_28d": "21-28 d",
    "young_49_52d": "49-52 d",
    "young_104_132d": "104-132 d",
}
COLOUR = {"control": pf.OKABE_ITO["blue"], "treated": pf.OKABE_ITO["vermillion"]}


def main():
    rows = list(csv.DictReader(open(HERE / "samples.csv")))
    pf.use_style()
    fig, ax = pf.figure(pf.ONE_COLUMN_MM, 62)
    rng = np.random.default_rng(0)
    for i, w in enumerate(WINDOW_LABELS):
        for j, t in enumerate(["control", "treated"]):
            ys = [
                int(r["trimmed_pairs"])
                for r in rows
                if r["window"] == w and r["treatment"] == t
            ]
            xs = i + (j - 0.5) * 0.36 + rng.uniform(-0.07, 0.07, len(ys))
            ax.scatter(
                xs,
                ys,
                s=10,
                color=COLOUR[t],
                edgecolor="none",
                label=f"{t} libraries" if i == 0 else None,
                zorder=3,
            )
            ax.text(
                i + (j - 0.5) * 0.36,
                1.3e8,
                str(len(ys)),
                ha="center",
                va="bottom",
                color="black",
            )
    ax.axhline(
        5e6, color=pf.GREY, lw=0.5, ls="--", label="5 million read pairs", zorder=1
    )
    ax.set_yscale("log")
    ax.set_ylim(100, 4e8)
    ax.set_yticks([1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8])
    ax.set_yticklabels(
        ["100", "1,000", "10,000", "100,000", "1 million", "10 million", "100 million"]
    )
    ax.set_xticks(range(len(WINDOW_LABELS)))
    ax.set_xticklabels(WINDOW_LABELS.values())
    ax.set_xlabel("Time after treatment (window)")
    ax.set_ylabel("Read pairs after trimming")
    ax.text(-0.6, 1.3e8, "n =", ha="right", va="bottom")
    pf.shared_legend(fig, ncol=3)
    out = HERE / "figures"
    out.mkdir(exist_ok=True)
    pf.save(fig, out / "design_libraries_per_window")


if __name__ == "__main__":
    main()
