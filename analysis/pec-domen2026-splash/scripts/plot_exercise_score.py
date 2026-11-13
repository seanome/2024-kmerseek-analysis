#!/usr/bin/env python3
"""Figures for the exercise score of Domen et al. 2026 Fig. 4C-D (notebook 361).

Usage: plot_exercise_score.py <exercise_score_comparisons.tsv> <exercise_score_samples.tsv> <out_dir>

exercise_score_recovery: a, b. Our AUC and p against the printed ones, one row per printed comparison.
  c. AUC with the groups in the authors' code and with the groups the panel's words describe.
exercise_score_samples: every sample's score in the 8 datasets, groups from the authors' code.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "notebooks"))
import pubfig as pf  # noqa: E402

comp_p, samp_p, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
c = pd.read_csv(comp_p, sep="\t")
samp = pd.read_csv(samp_p, sep="\t")
BLUE, ORANGE = pf.OKABE_ITO["blue"], pf.OKABE_ITO["orange"]
pf.use_style()
ROW = {
    "main": "",
    "pre_vs_30min": " 30 min",
    "pre_vs_3h": " 3 h",
    "pre_vs_24h": " 24 h",
    "full_model": " all clusters",
    "early_vs_rest": " Table S1",
}

fig = plt.figure(figsize=(pf.TWO_COLUMN_MM * pf.MM, 150 * pf.MM), layout="constrained")
top, bottom = fig.subfigures(2, 1, height_ratios=[1, 1.05])
axa, axb = top.subplots(1, 2, sharey=True)
t = c[c.published_auc.notna()].reset_index(drop=True)
labels = [f"{d}{ROW[n]}" for d, n in zip(t.dataset, t.comparison)]
y = np.arange(len(t))
for ax in (axa, axb):
    ax.grid(axis="y", color="#dddddd", linewidth=0.4)
    ax.set_axisbelow(True)
axa.scatter(
    t.published_auc, y, s=46, color=ORANGE, alpha=0.45, linewidths=0, label="printed in the paper or its Table S1"
)
axa.scatter(t.our_auc, y, s=9, color=BLUE, linewidths=0, label="ours, from GEO tables")
axa.axvline(0.5, color=pf.GREY, linewidth=0.6, linestyle=":")
axa.set_xlim(0, 1)
axa.set_xlabel("AUC (0.5: scores do not separate the groups)")
axa.set_yticks(y, labels)
axa.set_ylim(len(t) - 0.5, -0.5)
has_p = t.published_p.notna()
axb.scatter(t.published_p[has_p], y[has_p], s=46, color=ORANGE, alpha=0.45, linewidths=0)
axb.scatter(t.our_p, y, s=9, color=BLUE, linewidths=0)
for i in y[~has_p]:
    axb.annotate(
        "no p printed", (t.our_p[i], i), xytext=(6, 0), textcoords="offset points", ha="left", va="center", fontsize=5
    )
axb.set_xscale("log")
axb.set_xticks([1, 1e-3, 1e-6, 1e-9, 1e-12], ["1", "0.001", "1e-6", "1e-9", "1e-12"])
axb.minorticks_off()
axb.set_xlim(2, 3e-14)
axb.set_xlabel("p, Welch's t-test")
top.legend(*axa.get_legend_handles_labels(), loc="outside upper center", ncol=2, frameon=False)
pf.panel_label(axa, "a", dx_pt=-75)
pf.panel_label(axb, "b")

ax = bottom.subplots()
WORDS = {
    ("GSE104079", "main"): "GSE104079: fasting vs fasting + exercise, liver and soleus; or soleus, control vs exercise",
    ("GSE27536", "main"): "GSE27536: sedentary vs trained, healthy + COPD; or COPD only",
    ("GSE120862", "main"): "GSE120862: all untrained vs all trained; or before vs after one bout",
    ("GSE120862", "leg_after_bout"): "GSE120862: rested vs exercised leg, after one bout",
    ("GSE120862", "rest_after_training"): "GSE120862: rested leg at rest, before vs after training",
    ("GSE83578", "pre_vs_30min"): "GSE83578, 30 min: control + LPS blood; or control blood",
    ("GSE83578", "pre_vs_3h"): "GSE83578, 3 h: control + LPS blood; or control blood",
    ("GSE83578", "pre_vs_24h"): "GSE83578, 24 h: control + LPS blood; or control blood",
    ("GSE99963", "main"): "GSE99963: before vs after, both arms; or exercise arm",
    ("GSE144304", "main"): "GSE144304: young vs fit old; or frail old vs fit old",
}
for i, key in enumerate(WORDS):
    r = c[(c.dataset == key[0]) & (c.comparison == key[1])].set_index("grouping")
    ax.axhline(i, color="#dddddd", linewidth=0.4, zorder=0)
    if "authors" in r.index:
        ax.plot(r.loc["authors", "our_auc"], i, "o", color=BLUE, markersize=4, zorder=3)
    ax.plot(
        r.loc["as_labelled", "our_auc"],
        i,
        "o",
        markerfacecolor="white",
        color=BLUE,
        markersize=4,
        markeredgewidth=0.8,
        zorder=2,
    )
ax.axvline(0.5, color=pf.GREY, linewidth=0.6, linestyle=":")
ax.set_yticks(range(len(WORDS)), list(WORDS.values()))
ax.set_ylim(len(WORDS) - 0.5, -0.5)
ax.set_xlim(0, 1)
ax.set_xlabel("AUC of the second group against the first (0.5: scores do not separate the groups)")
bottom.legend(
    handles=[
        Line2D([], [], linestyle="none", marker="o", color=BLUE, markersize=4, label="groups in the authors' code"),
        Line2D(
            [],
            [],
            linestyle="none",
            marker="o",
            markerfacecolor="white",
            color=BLUE,
            markersize=4,
            markeredgewidth=0.8,
            label="groups closer to the panel's words",
        ),
    ],
    loc="outside upper center",
    ncol=2,
    frameon=False,
)
pf.panel_label(ax, "c", dx_pt=-200)
pf.save(fig, str(out / "exercise_score_recovery"))

PANELS = [
    ("GSE104079", ["main"], "mice, liver + soleus", ["fasting", "fasting +\nexercise"]),
    ("GSE27536", ["main"], "healthy + COPD, cluster 13", ["sedentary", "trained"]),
    ("GSE120862", ["main"], "all 84 samples", ["untrained", "trained"]),
    ("GSE83578", ["pre_vs_30min", "pre_vs_3h", "pre_vs_24h"], "blood, control + LPS", ["0", "30 min", "3 h", "24 h"]),
    ("GSE111555", ["main"], "PBMC", ["before", "after"]),
    ("GSE99963", ["main"], "both arms", ["before", "after"]),
    ("GSE144304", ["main"], "ages 20-27 and 75-85", ["young", "fit old"]),
    ("GSE28998", ["main"], "biceps, 4 h after", ["untrained\narm", "trained\narm"]),
]
fig, axs = pf.figure(pf.TWO_COLUMN_MM, 105, nrows=2, ncols=4, layout="constrained")
rng = np.random.default_rng(361)
for ax, (d, names, title, xt) in zip(axs.flat, PANELS):
    s = samp[(samp.dataset == d) & (samp.grouping == "authors") & samp.comparison.isin(names)]
    groups = [s[(s.comparison == names[0]) & (s.group == 0)].score.to_numpy()]
    groups += [s[(s.comparison == n) & (s.group == 1)].score.to_numpy() for n in names]
    for i, v in enumerate(groups):
        ax.scatter(i + rng.uniform(-0.18, 0.18, len(v)), v, s=5, color=BLUE, linewidths=0, alpha=0.8)
        ax.plot([i - 0.28, i + 0.28], [np.median(v)] * 2, color="black", linewidth=0.8)
    ax.set_xticks(range(len(groups)), [f"{a}\nn = {len(v)}" for a, v in zip(xt, groups)])
    ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.set_title(f"{d}\n{title}")
axs[0, 0].set_ylabel("score")
axs[1, 0].set_ylabel("score")
fig.suptitle(
    "Each dot is one sample, black line the median. Higher: more cell-repair genes; lower: more pro-immune genes",
    fontsize=6,
)
pf.save(fig, str(out / "exercise_score_samples"))
print("rows", len(t), "panels", len(PANELS))
