"""Draw projection_scores.csv: Fmax per species, as the pipeline scores it and with each domain placed.

Run: python projection_figure.py   (reads projection_scores.csv beside this file)
"""

import sys
from pathlib import Path

import numpy as np
import polars as pl

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parents[1] / "notebooks"))
import pubfig  # noqa: E402

sys.path.insert(0, str(HERE.parents[1] / "scripts"))
import shrink_png  # noqa: E402

s = pl.read_csv(HERE / "projection_scores.csv")
# Same filter as projection_test.summarise(): settings with calls in at least 5 species.
k = s.filter(pl.col("search") != "mmseqs2", pl.col("n_calls") > 0)
keep = (
    k.group_by("search", "arm", "stratum").len().filter(pl.col("len") >= 5).drop("len")
)
k = (
    k.join(keep, on=["search", "arm", "stratum"])
    .group_by("species", "arm", "stratum")
    .agg(pl.col("fmax").median())
    .with_columns(pl.lit("kmerseek").alias("tool"))
)
mm = s.filter(pl.col("search") == "mmseqs2").select(
    "species", "arm", "stratum", "fmax", pl.lit("mmseqs2").alias("tool")
)
d = pl.concat([k, mm.select(k.columns)])

ARMS = [
    ("span", "As the pipeline\nscores it"),
    ("projected", "Each domain at its\nown place on the query"),
]
CUTS = [("all", "All proteins"), ("zinc_finger_c2h2", "C2H2 zinc-finger proteins")]
TOOLS = [
    ("mmseqs2", "MMseqs2", pubfig.OKABE_ITO["vermillion"], 0.13),
    ("kmerseek", "kmerseek, typical setting", pubfig.OKABE_ITO["blue"], -0.13),
]

pubfig.use_style()
fig, axs = pubfig.figure(width_mm=pubfig.TWO_COLUMN_MM, height_mm=45, ncols=2)
rng = np.random.default_rng(0)
for ax, (cut, title), letter in zip(axs, CUTS, "ab"):
    for i, (arm, _) in enumerate(reversed(ARMS)):
        for tool, label, colour, dy in TOOLS:
            v = d.filter(
                pl.col("stratum") == cut, pl.col("arm") == arm, pl.col("tool") == tool
            )["fmax"].to_numpy()
            ax.scatter(
                v,
                i + dy + rng.uniform(-0.04, 0.04, len(v)),
                s=8,
                color=colour,
                alpha=0.7,
                linewidths=0,
                label=label if (ax is axs[0] and i == 0) else None,
            )
            ax.plot(
                [np.median(v)] * 2, [i + dy - 0.09, i + dy + 0.09], color=colour, lw=1
            )
    ax.set_yticks(range(len(ARMS)), [a[1] for a in reversed(ARMS)])
    ax.set_ylim(-0.5, len(ARMS) - 0.5)
    ax.set_xlim(0, 0.8)
    ax.set_xlabel(
        f"Fmax, {title.lower().replace('c2h2', 'C2H2')}\n(one dot per species, bar = median)"
    )
    pubfig.panel_label(ax, letter)
pubfig.shared_legend(fig)
fig.savefig(HERE / "projection.png", dpi=300)
shrink_png.process(HERE / "projection.png", check=False)
fig.savefig(HERE / "projection.pdf")
