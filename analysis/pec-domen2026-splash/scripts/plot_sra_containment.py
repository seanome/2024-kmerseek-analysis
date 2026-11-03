#!/usr/bin/env python3
"""Containment of the paper's list proteins in each searched run.

Usage: plot_sra_containment.py <gene_run_containment.tsv.gz> <containment_by_setting.tsv> <cutoff> <out_stem> [frame_label]

frame_label describes how the reads' frames were chosen, for the heat map title. Settings are
read from containment_by_setting.tsv (every alphabet, k and cutoff in it, in that order).

a: heat map, HP k30, at <cutoff>: one row per list protein (grouped by list direction, then
   contrast), one column per Botryllus run (grouped by contrast, then control before PEC).
b: per setting (alphabet, k, cutoff), median containment over protein-run pairs in Botryllus
   runs, for list proteins and length-matched random proteins; human runs alongside (open).
c: sanity check per contrast: mean containment of the contrast's "up" and "down" proteins in
   its PEC runs minus its control runs (HP k30). Containment is presence, not abundance, and
   reads per run differ, so this is only a direction check.
Writes <out_stem>.tsv with the numbers in b and c.
"""

import sys
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import polars as pl
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "notebooks"))
import pubfig as pf  # noqa: E402

tab_p, summ_p, cutoff, out_stem = sys.argv[1:5]  # summ_p: containment_by_setting.tsv
FRAME_LABEL = (
    sys.argv[5]
    if len(sys.argv) > 5
    else f"reads whose frame has >= {cutoff} of its 7-mers in Swiss-Prot"
)
d = pl.read_csv(
    tab_p,
    separator="\t",
    infer_schema_length=10000,
    schema_overrides={"cutoff": pl.Utf8},
)
summ = pl.read_csv(summ_p, separator="\t", schema_overrides={"cutoff": pl.Utf8})
CONTRASTS = [
    "young_2h",
    "young_24h",
    "old_24h",
    "young_13_49d",
    "young_132d",
    "old_41_93d",
    "not_in_SI",
]
NAME = {
    "young_2h": "young 2 h",
    "young_24h": "young 24 h",
    "old_24h": "old 24 h",
    "young_13_49d": "young 13-49 d",
    "young_132d": "young 132 d",
    "old_41_93d": "old 41-93 d",
    "not_in_SI": "not in the paper's SI",
}
# reddish purple means KaMRaT elsewhere in this PR, so list proteins are sky blue here
LIST_COL, RAND_COL = pf.OKABE_ITO["sky_blue"], "black"
PEC_COL, CONTROL_COL = "black", pf.GREY

bot = d.filter(
    (pl.col("dataset") == "PRJNA1219309")
    & (pl.col("alphabet") == "hp")
    & (pl.col("k") == 30)
)
lists = bot.filter(pl.col("set") == "paper_list")
# run order: contrast, then control before PEC, then sample
runs = (
    bot.select("run", "sample", "condition", "run_contrast")
    .unique()
    .with_columns(
        o1=pl.col("run_contrast").replace_strict(
            {c: i for i, c in enumerate(CONTRASTS)}, default=99
        ),
        o2=(pl.col("condition") == "PEC").cast(pl.Int8),
    )
    .sort("o1", "o2", "sample")
)
# protein order: first list direction, then first list contrast, then mean containment
prot = (
    lists.group_by("query", "paper_list")
    .agg(pl.col("containment").mean().alias("mean_c"))
    .with_columns(first=pl.col("paper_list").str.split(";").list.first())
    .with_columns(
        direction=pl.col("first").str.split(":").list.get(1),
        contrast=pl.col("first").str.split(":").list.get(0),
    )
    .with_columns(
        o1=(pl.col("direction") == "down").cast(pl.Int8),
        o2=pl.col("contrast").replace_strict(
            {c: i for i, c in enumerate(CONTRASTS)}, default=99
        ),
    )
    .sort("o1", "o2", "mean_c", descending=[False, False, True])
)
mat = lists.pivot(
    on="run", index="query", values="containment", aggregate_function="first"
)
mat = (
    prot.select("query")
    .join(mat, on="query", how="left")
    .select(["query"] + runs["run"].to_list())
)
M = mat.drop("query").to_numpy().astype(float)

pf.use_style()
fig = plt.figure(figsize=(pf.TWO_COLUMN_MM * pf.MM, 150 * pf.MM), layout="constrained")
gs = fig.add_gridspec(3, 2, height_ratios=[2.2, 0.05, 1], width_ratios=[1.3, 1])
a = fig.add_subplot(gs[0, :])
strip = fig.add_subplot(gs[1, :], sharex=a)
b = fig.add_subplot(gs[2, 0])
c = fig.add_subplot(gs[2, 1])

cmap = sns.color_palette("mako_r", as_cmap=True).copy()
cmap.set_bad("#FFFFFF")
im = a.imshow(
    M, aspect="auto", cmap=cmap, norm=mcolors.Normalize(0, 1), interpolation="nearest"
)
cb = fig.colorbar(im, ax=[a, strip], fraction=0.025, pad=0.01)
cb.set_label("Containment of the protein's HP k30 hashes in the run")
# column groups
rr = runs.to_dicts()
bounds, ticks, labels = [], [], []
start = 0
for i in range(1, len(rr) + 1):
    if i == len(rr) or rr[i]["run_contrast"] != rr[start]["run_contrast"]:
        ticks.append((start + i - 1) / 2)
        labels.append(NAME.get(rr[start]["run_contrast"], rr[start]["run_contrast"]))
        if i < len(rr):
            bounds.append(i - 0.5)
        start = i
for x in bounds:
    a.axvline(x, color="white", lw=1.2)
    strip.axvline(x, color="white", lw=1.2)
a.tick_params(axis="x", length=0, labelbottom=False)
# condition strip under the columns: grey = control, black = PEC
cond = np.array([[0 if r["condition"] == "control" else 1 for r in rr]])
strip.imshow(
    cond,
    aspect="auto",
    cmap=mcolors.ListedColormap([CONTROL_COL, PEC_COL]),
    vmin=0,
    vmax=1,
    interpolation="nearest",
)
strip.set_yticks([])
strip.set_xticks(ticks, labels)
strip.tick_params(axis="x", length=0)
for side in ["left", "bottom"]:
    strip.spines[side].set_visible(False)
# row groups
pr = prot.to_dicts()
start = 0
yt, yl = [], []
for i in range(1, len(pr) + 1):
    if i == len(pr) or (pr[i]["direction"], pr[i]["contrast"]) != (
        pr[start]["direction"],
        pr[start]["contrast"],
    ):
        yt.append((start + i - 1) / 2)
        yl.append(
            f"{NAME.get(pr[start]['contrast'])} {pr[start]['direction']} ({i - start})"
        )
        if i < len(pr):
            a.axhline(
                i - 0.5,
                color="white",
                lw=0.8 if pr[i]["direction"] == pr[start]["direction"] else 2,
            )
        start = i
a.set_yticks(yt, yl)
a.tick_params(axis="y", length=0)
strip.set_xlabel(
    "Botryllus run, grouped by contrast (strip: grey = control, black = PEC)"
)
a.set_ylabel("List protein (first list it is on, number of proteins)")
a.set_title(f"HP k30, {FRAME_LABEL}", loc="left")

# b: alphabets and k sizes
rows = []
settings = summ.select("alphabet", "k").unique(maintain_order=True).rows()
cuts = sorted(summ["cutoff"].unique())
xpos = 0
xt, xl = [], []
for alpha, k in settings:
    for cut in cuts:
        for sp, marker in [("Botryllus", "o"), ("human", "^")]:
            for st, col in [
                ("paper_list", LIST_COL),
                ("random_length_matched", RAND_COL),
            ]:
                y = summ.filter(
                    (pl.col("alphabet") == alpha)
                    & (pl.col("k") == k)
                    & (pl.col("cutoff") == cut)
                    & (pl.col("species") == sp)
                    & (pl.col("set") == st)
                )
                if y.height == 0:
                    continue
                y = y.row(0, named=True)
                b.scatter(
                    xpos + (0.12 if st == "random_length_matched" else -0.12),
                    y["median_containment"],
                    marker=marker,
                    s=10,
                    lw=0.6,
                    edgecolors=col,
                    facecolors=col if sp == "Botryllus" else "white",
                )
                rows.append(
                    {
                        "panel": "b",
                        "alphabet": alpha,
                        "k": k,
                        "cutoff": cut,
                        "species": sp,
                        "set": st,
                        "pairs": y["pairs"],
                        "median_containment": round(y["median_containment"], 4),
                        "mean_containment": round(y["mean_containment"], 4),
                        "share_pairs_any_hash": round(y["share_any_hash"], 4),
                    }
                )
        xt.append(xpos)
        name = f"{'HP' if alpha == 'hp' else alpha.capitalize()} {k}"
        xl.append(name if len(cuts) == 1 else f"{name}, >= {cut}")
        xpos += 1
b.set_xticks(xt, xl, rotation=45, ha="right")
b.set_ylabel("Median containment (protein-run pairs)")
b.scatter(
    [], [], color=LIST_COL, marker="o", s=10, label="list proteins, Botryllus runs"
)
b.scatter(
    [], [], color=RAND_COL, marker="o", s=10, label="random proteins, Botryllus runs"
)
b.scatter(
    [],
    [],
    facecolors="white",
    edgecolors=LIST_COL,
    marker="^",
    s=10,
    lw=0.6,
    label="list proteins, human runs",
)
b.scatter(
    [],
    [],
    facecolors="white",
    edgecolors=RAND_COL,
    marker="^",
    s=10,
    lw=0.6,
    label="random proteins, human runs",
)
b.legend(loc="upper right", fontsize=5)

# c: up vs down, PEC minus control, per contrast
ex = []
for i, ctr in enumerate(CONTRASTS[:-1]):
    for direction, dx in [("up", -0.12), ("down", 0.12)]:
        q = lists.filter(
            pl.col("paper_list").str.contains(f"{ctr}:{direction}")
            & (pl.col("run_contrast") == ctr)
        )
        if q.height == 0:
            continue
        pec = q.filter(pl.col("condition") == "PEC")["containment"].mean()
        con = q.filter(pl.col("condition") == "control")["containment"].mean()
        diff = (pec or 0) - (con or 0)
        c.bar(
            i + dx,
            diff,
            width=0.22,
            color=LIST_COL if direction == "up" else "white",
            edgecolor=LIST_COL,
            lw=0.6,
        )
        ex.append(
            {
                "panel": "c",
                "contrast": ctr,
                "direction": direction,
                "proteins": q["query"].n_unique(),
                "mean_containment_PEC": round(pec, 4) if pec is not None else None,
                "mean_containment_control": round(con, 4) if con is not None else None,
                "PEC_minus_control": round(diff, 4),
            }
        )
c.axhline(0, color="black", lw=0.5)
c.set_xticks(
    range(len(CONTRASTS) - 1),
    [NAME[x] for x in CONTRASTS[:-1]],
    rotation=30,
    ha="right",
)
c.set_ylabel("Mean containment, PEC - control")
c.legend(
    handles=[
        Patch(facecolor=LIST_COL, edgecolor=LIST_COL, label="paper's up list"),
        Patch(facecolor="white", edgecolor=LIST_COL, label="paper's down list"),
    ],
    loc="upper right",
    fontsize=5,
)
for ax, letter in zip([a, b, c], "abc"):
    pf.panel_label(ax, letter)
pl.DataFrame(rows + ex, infer_schema_length=None).write_csv(
    f"{out_stem}.tsv", separator="\t"
)
pf.save(fig, out_stem)
print(pl.DataFrame(ex))
