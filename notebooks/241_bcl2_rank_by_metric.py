#!/usr/bin/env python3
"""Where BCL2 ranks among the human proteins Ced-9 hits, under four metrics.

One dot per arm (alphabet and k) that hit BCL2, from ranks.csv of the 241 sweep. Panel A is
the rank itself. Panel B divides it by the number of proteins that arm hit, because the
arms hit anywhere from about 1_000 to 18_000 proteins: rank 950 is near the top of 18_000
and near the bottom of 1_161. In panel B, 0.5 is where a randomly chosen hit protein lands
on average.

Writes figures/241_bcl2_rank_by_metric.png and prints the table it draws from.

Run with the 2025-kmerseek-analysis env:
    python notebooks/241_bcl2_rank_by_metric.py
"""

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FixedLocator, NullLocator  # noqa: E402

# NB241_DIR is set by 241_alphabet_ranking.sbatch on Sherlock; the default is the laptop
# folder that holds the small tables copied back for this notebook.
DATA = Path(os.environ.get("NB241_DIR", "/Users/olga/data/botryllus/alphabet-ranking-three-cases"))
FIG = Path(__file__).resolve().parent.parent / "figures"
METRICS = ["E-value", "Poisson score", "tf-idf", "mean IDF"]
ARM_COLOR, MARK_COLOR = "#4C72B0", "#C44E52"

pl.Config.set_tbl_width_chars(200)
r = pl.read_csv(DATA / "ranks.csv", infer_schema_length=None).filter(
    (pl.col("query") == "Ced9") & pl.col("metric").is_in(METRICS)
)
found = r.filter(pl.col("partner_found")).with_columns(
    (pl.col("rank") / pl.col("n_targets")).alias("share")
)
n_arms = r.filter(pl.col("metric") == "mean IDF").height
n_hit = found.filter(pl.col("metric") == "mean IDF").height

table = (
    found.group_by("metric")
    .agg(
        pl.len().alias("n_arms_ranked"),
        pl.col("rank").min().alias("best_rank"),
        pl.col("rank").median().alias("median_rank"),
        pl.col("share").min().round(3).alias("best_share"),
        pl.col("share").median().round(3).alias("median_share"),
        (pl.col("share") < 0.5).sum().alias("n_arms_top_half"),
        pl.col("n_tied").max().alias("max_tied"),
    )
    .with_columns(pl.col("metric").replace_strict({m: i for i, m in enumerate(METRICS)}).alias("o"))
    .sort("o")
    .drop("o")
)
print(f"{n_hit} of {n_arms} arms hit BCL2")
print(table)

fig, axes = plt.subplots(1, 2, figsize=(14, 6.6), sharex=True)
rng = np.random.default_rng(0)
jitter = {m: rng.uniform(-0.18, 0.18, found.filter(pl.col("metric") == m).height) for m in METRICS}
ticks = []
for i, m in enumerate(METRICS):
    d = found.filter(pl.col("metric") == m)
    lab = f"{m}\nBCL2 ranked in\n{d.height} of {n_arms} arms"
    if d.height < n_hit:
        lab += f"\n({n_hit - d.height} hit it\nwith no E-value)"
    ticks.append(lab)
    k17 = d.filter((pl.col("alphabet") == "hp_lehninger2") & (pl.col("ksize") == 17))
    for ax, col, text_y in ((axes[0], "rank", 12), (axes[1], "share", -0.04)):
        x = i + jitter[m]
        ax.scatter(x, d[col], s=28, color=ARM_COLOR, alpha=0.75, zorder=3)
        ax.hlines(d[col].median(), i - 0.3, i + 0.3, color="black", lw=2.2, zorder=4)
        if k17.height:
            ax.scatter([i + 0.36], k17[col], marker="D", s=46, facecolor="white",
                       edgecolor=MARK_COLOR, lw=1.8, zorder=5)  # fmt: skip
        j = int(np.argmin(d[col].to_numpy()))
        b = d.row(j, named=True)
        val = f"{b['rank']:,}" if col == "rank" else f"{b['rank']:,} of {b['n_targets']:,}"
        ax.annotate(f"best {val}\n{b['alphabet']} k={b['ksize']}", (x[j], b[col]),
                    xytext=(i, text_y), ha="center", va="bottom", fontsize=8.5,
                    arrowprops=dict(arrowstyle="-", color="grey", lw=0.7))  # fmt: skip
print("points per panel:", found.height)

ax = axes[0]
ax.set_yscale("log")
ax.set_ylim(20_000, 1)
ax.yaxis.set_major_locator(FixedLocator([1, 10, 100, 1_000, 10_000]))
ax.yaxis.set_minor_locator(NullLocator())
ax.set_yticklabels(["1", "10", "100", "1,000", "10,000"])
ax.set_ylabel("BCL2's rank among the human proteins that arm hit\n(1 = top; log scale)")
ax.set_title("A. Rank", loc="left", fontweight="bold")

ax = axes[1]
ax.axhline(0.5, color="grey", ls="--", lw=1.2, zorder=1)
ax.set_ylim(1, -0.16)  # room above 0 for the labels, clear of the dots near the top
ax.yaxis.set_major_locator(FixedLocator([0, 0.2, 0.4, 0.6, 0.8, 1.0]))
ax.set_ylabel("BCL2's rank divided by the number of proteins that arm hit\n(0 = top, 1 = bottom)")
ax.set_title("B. Rank as a share of the proteins hit", loc="left", fontweight="bold")

for ax in axes:
    ax.set_xticks(range(len(METRICS)), ticks, fontsize=9)
    ax.set_xlim(-0.6, len(METRICS) - 0.4)
    ax.set_xlabel("metric used to sort the hits")
    ax.grid(axis="y", color="#ddd", zorder=0)

handles = [
    Line2D([], [], marker="o", ls="", color=ARM_COLOR, label="one alphabet and k (arm)"),
    Line2D([], [], color="black", lw=2.2, label="median over arms"),
    Line2D([], [], marker="D", ls="", mfc="white", mec=MARK_COLOR, mew=1.8, label="hp_lehninger2 k=17"),
    Line2D([], [], color="grey", ls="--", lw=1.2, label="a randomly chosen hit protein, on average (B)"),
]
fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.925), fontsize=9.5)
fig.suptitle(
    "Under every metric, BCL2 ranks below the typical protein Ced-9 hits in the human proteome",
    fontsize=13, fontweight="bold", y=0.985,
)  # fmt: skip
fig.text(
    0.5, 0.945,
    "Ced-9 searched against 19_732 GENCODE human proteins (notebook 241). "
    "Only arms that hit BCL2 are drawn.",
    ha="center", fontsize=9, color="#444",
)  # fmt: skip
fig.tight_layout(rect=(0, 0, 1, 0.9))
out = FIG / "241_bcl2_rank_by_metric.png"
fig.savefig(out, dpi=150)
print(f"wrote {out}")
