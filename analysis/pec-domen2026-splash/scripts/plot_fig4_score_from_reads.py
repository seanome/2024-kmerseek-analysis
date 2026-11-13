#!/usr/bin/env python3
"""Figure: the Fig. 4C-D score from reads, beside the score from the authors' gene tables (notebook 361).

Usage: plot_fig4_score_from_reads.py <fig4_reads_comparisons.tsv> <fig4_reads_samples.tsv>
           <fig4_reads_random.tsv.gz> <out_stem> [<gene_table_check/fig4_reads_comparisons.tsv>]

One column per RNA-seq series of Fig. 4C-D. Row a: each sample's score from the gene table (PR 212).
Row b: the same samples' score from reads counted with no genome, at the largest share of reads
counted. Row c: the AUC at each share of reads, against the printed AUC and against random gene sets.
With the fifth argument, row c also shows the authors' gene table thinned to the same number of reads.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "notebooks"))
import pubfig as pf  # noqa: E402

comp_tsv, samples_tsv, random_tsv, out_stem = sys.argv[1:5]
table = pd.read_csv(sys.argv[5], sep="\t") if len(sys.argv) > 5 else None
comp = pd.read_csv(comp_tsv, sep="\t")
samples = pd.read_csv(samples_tsv, sep="\t")
rand = pd.read_csv(random_tsv, sep="\t")
C = pf.OKABE_ITO
FIRST, SECOND = C["blue"], C["vermillion"]  # the first and second group of every series
PANELS = {
    "GSE120862": "GSE120862 (panel C)\nuntrained vs trained",
    "GSE99963": "GSE99963 (panel D)\nbefore vs after 10 days",
    "GSE144304": "GSE144304 (panel D)\nyoung vs fit old adults",
}
rng = np.random.default_rng(0)  # only for the sideways spread of the dots


def reads(v):
    return f"{v / 1e6:.1f} M" if v >= 1e6 else f"{v / 1e3:.0f} k"


pf.use_style()
fig, axs = pf.figure(pf.TWO_COLUMN_MM, 125, nrows=3, ncols=3, layout="constrained")


def strip(ax, d, column):
    """One dot per sample, the two groups on two rows, a tick at each group's median."""
    groups = list(dict.fromkeys(d.group))
    for y, (g, colour) in enumerate(zip(groups, (FIRST, SECOND))):
        v = d.loc[d.group == g, column].to_numpy()
        ax.scatter(v, y + rng.uniform(-0.22, 0.22, len(v)), s=7, color=colour, linewidths=0, alpha=0.8)
        ax.plot([np.median(v)] * 2, [y - 0.36, y + 0.36], color="black", lw=1)
    ax.set_yticks(range(len(groups)), [f"{g} (n = {(d.group == g).sum()})" for g in groups])
    ax.set_ylim(len(groups) - 0.4, -0.6)


for j, (series, title) in enumerate(PANELS.items()):
    c = comp[comp.series == series].sort_values("share")
    if c.empty:
        for ax in axs[:, j]:
            ax.set_axis_off()
        continue
    most = c.share.min()  # share 1 = every read; the smallest number is the largest share counted
    d = samples[(samples.series == series) & (samples.share == most)]
    top = c[c.share == most].iloc[0]

    ax = axs[0, j]
    ax.set_title(title)
    strip(ax, d, "score_from_gene_table")
    ax.set_xlabel(f"Score from the gene table (PR 212)\nprinted AUC {top.printed_auc:.2f}")

    ax = axs[1, j]
    strip(ax, d, "score_from_reads")
    ax.set_xlabel(f"Score from reads, no genome\n{top.share_of_reads} reads, AUC {top.auc_from_reads:.2f}")

    ax = axs[2, j]
    x = c.median_reads_assigned_per_sample.to_numpy()
    r = rand[rand.series == series].groupby("share").auc.quantile([0.025, 0.5, 0.975]).unstack().loc[c.share]
    ax.axhspan(
        top.printed_auc - 0.15,
        top.printed_auc + 0.15,
        color=C["yellow"],
        alpha=0.35,
        lw=0,
        label="within 0.15 of the printed AUC (rule 1)",
    )
    ax.axhline(top.printed_auc, color="black", lw=0.8, ls="--", label="printed AUC")
    ax.axhline(0.5, color=pf.GREY, lw=0.5, ls=":", label="0.5, no difference")
    ax.errorbar(
        x * 1.12,
        r[0.5],
        yerr=[r[0.5] - r[0.025], r[0.975] - r[0.5]],
        fmt="x",
        color=pf.GREY,
        ms=3,
        lw=0.8,
        capsize=1.5,
        label="1,000 random gene sets: median, middle 95 %",
    )
    if table is not None and (table.series == series).any():
        t = table[table.series == series].sort_values("share")
        ax.plot(
            t.median_reads_assigned_per_sample,
            t.auc_from_reads,
            color=C["reddish_purple"],
            lw=1,
            ls="--",
            marker="s",
            ms=3,
            zorder=2,
            label="the authors' gene table, thinned to as many reads",
        )
    ok = c.rule1_printed_auc_comes_back.to_numpy()
    ax.plot(x, c.auc_from_reads, color=C["bluish_green"], lw=1, zorder=3)
    ax.scatter(x[ok], c.auc_from_reads[ok], s=16, color=C["bluish_green"], zorder=4, label="from reads, rule 1 passes")
    ax.scatter(
        x[~ok],
        c.auc_from_reads[~ok],
        s=16,
        facecolor="white",
        edgecolor=C["bluish_green"],
        zorder=4,
        label="from reads, rule 1 fails",
    )
    ax.set_xscale("log")
    ax.set_xticks(x, [f"{reads(v)}\n({s})" for v, s in zip(x, c.share_of_reads)])
    if table is not None and (table.series == series).any():
        ax.set_xlim(x.min() / 1.6, max(x.max(), t.median_reads_assigned_per_sample.max()) * 1.6)
    ax.minorticks_off()
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("Reads placed in a gene per sample\n(median; share of reads counted)")
    if j == 0:
        ax.set_ylabel("AUC, second group above first")

handles, labels = {}, {}
for ax in axs[2]:  # a series can lack one of the marks, so collect them from every panel
    for h, l in zip(*ax.get_legend_handles_labels()):
        handles.setdefault(l, h)
handles, labels = list(handles.values()), list(handles)
fig.legend(handles, labels, loc="outside upper center", ncol=3, frameon=False)
for ax, letter in zip(axs[:, 0], "abc"):
    pf.panel_label(ax, letter, dx_pt=-40)
pf.save(fig, out_stem)
