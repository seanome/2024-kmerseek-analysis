#!/usr/bin/env python3
"""Botryllus routes scored against the paper's lists and against nf-core's lists (notebook 361).

Usage: plot_nfcore_recovery.py <paper_recovery.tsv> <gene_counts_paper.tsv> <gene_counts_mean_paper.tsv> <nfcore recovery.tsv> <list_sizes.tsv> <out_stem>

a, b. Share of list genes reached in the list's direction (SPLASH has no direction). One small mark per
      list; filled if the list beat 1,000 random gene sets, open if not. One vertical bar per route:
      all lists together. a: the paper's 12 lists. b: the 12 nf-core lists at the paper's cutoff.
c.    Genes tested and genes with p <= 0.05 and 2-fold in each contrast, with the count chance alone
      would give at p <= 0.05 (5 % of genes tested).
Writes <out_stem>.tsv with the totals drawn, and prints it.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "notebooks"))
import pubfig as pf  # noqa: E402

paper_tsv, counts_tsv, mean_tsv, nfcore_tsv, sizes_tsv, out_stem = sys.argv[1:]
C = pf.OKABE_ITO
ROUTES = {  # channel: (label, colour, marker)
    "gene_counts_mean": ("gene-counts, mean of samples", C["black"], "s"),
    "gene_counts": ("gene-counts, summed reads", C["black"], "D"),
    "splash": ("SPLASH", C["blue"], "o"),
    "anchor_nominal": ("anchor counts, p < 0.05", C["vermillion"], "s"),
    "kamrat_ttest_p05": ("KaMRaT, t-test, p < 0.05", C["reddish_purple"], "o"),
    "kamrat_deseq2_p05": ("KaMRaT, DESeq2, p < 0.05", C["reddish_purple"], "s"),
    "dekupl_limma_p05": ("DE-kupl, limma, p < 0.05", C["bluish_green"], "o"),
    "dekupl_deseq2_p05": ("DE-kupl, DESeq2, p < 0.05", C["orange"], "s"),
}
paper = pd.concat([pd.read_csv(paper_tsv, sep="\t"), pd.read_csv(counts_tsv, sep="\t").assign(channel="gene_counts"),
                   pd.read_csv(mean_tsv, sep="\t").assign(channel="gene_counts_mean")])
paper["genes"] = paper.list_genes_with_protein
nf = pd.read_csv(nfcore_tsv, sep="\t").query("cutoff == 'paper_cutoff'")
nf["genes"] = nf.list_genes


def totals(d):
    out = {}
    for ch in ROUTES:
        r = d[(d.channel == ch) & (d.genes > 0)]
        out[ch] = dict(genes=int(r.genes.sum()), reached=int(r.reached.sum()), share=r.reached.sum() / r.genes.sum(),
                       lists=len(r), lists_beat_random=int((r.share_null_at_or_above <= 0.05).sum()),
                       genes_given_this_direction=int((r.pool_fraction_reached * r.pool_genes).round().sum()))
    return pd.DataFrame(out).T


tp, tn = totals(paper), totals(nf)
t = pd.concat([tp.assign(reference="paper"), tn.assign(reference="nf-core")]).rename_axis("channel").reset_index()
t.to_csv(f"{out_stem}.tsv", sep="\t", index=False)
print(t.to_string(index=False))

pf.use_style()
fig, axs = pf.figure(pf.TWO_COLUMN_MM, 86, ncols=3, layout="constrained", gridspec_kw=dict(width_ratios=[1, 1, 0.9]))
ys = list(range(len(ROUTES)))[::-1]
rng = np.random.default_rng(361)
for ax, d, tot, title in ((axs[0], paper, tp, "the paper's 12 lists"), (axs[1], nf, tn, "12 nf-core DESeq2 lists, p ≤ 0.05, 2-fold")):
    for y, (ch, (lab, col, mk)) in zip(ys, ROUTES.items()):
        ax.axhline(y, color="0.9", linewidth=0.5, zorder=0)
        r = d[(d.channel == ch) & (d.genes > 0)]
        win = (r.share_null_at_or_above <= 0.05).to_numpy()
        jit = rng.uniform(-0.22, 0.22, len(r))
        ax.scatter(r.fraction_reached, y + jit, s=7, marker=mk, facecolors=[col if w else "white" for w in win], edgecolors=col,
                   linewidths=0.5, zorder=2, clip_on=False)
        ax.plot([tot.share[ch]] * 2, [y - 0.36, y + 0.36], color=col, linewidth=1.6, zorder=3)
        ax.annotate(f"{tot.reached[ch]:.0f} of {tot.genes[ch]:.0f}, {tot.lists_beat_random[ch]:.0f} of {tot.lists[ch]:.0f} lists", (1.0, y + 0.42), ha="right",
                    va="center", fontsize=5, color=col, annotation_clip=False)
    ax.axvline(0.5, color="0.45", linewidth=0.6, linestyle=":", zorder=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, len(ROUTES) - 0.3)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0", "25", "50", "75", "100"])
    ax.set_xlabel("list genes reached, going the same way (%)")
    ax.set_title(title)
axs[0].set_yticks(ys, [v[0] for v in ROUTES.values()])
axs[1].set_yticks(ys, [])

s = pd.read_csv(sizes_tsv, sep="\t")
NAMES = {"young_2h": "young 2 h", "young_24h": "young 24 h", "old_24h": "old 24 h", "young_13_49d": "young 13-49 d",
         "young_132d": "young 132 d", "old_41_93d": "old 41-93 d"}
ax = axs[2]
for _, r in s.iterrows():
    ax.plot(r.genes_tested, r.genes_tested * 0.05, marker="_", markersize=7, color="0.45", linestyle="none", zorder=2)
    ax.plot(r.genes_tested, r.p05_2fold, marker="o", markersize=3.5, color="0.15", linestyle="none", zorder=3)
    ax.annotate(NAMES[r.contrast], (r.genes_tested, r.p05_2fold), xytext=(0, 5 if r.contrast == "old_41_93d" else -7), textcoords="offset points",
                ha="center", fontsize=5)
ax.plot([], [], marker="_", markersize=7, color="0.45", linestyle="none", label="expected by chance at p ≤ 0.05 (5 % of genes)")
ax.plot([], [], marker="o", markersize=3.5, color="0.15", linestyle="none", label="genes with p ≤ 0.05 and 2-fold")
ax.set_yscale("log")
ax.set_yticks([10, 100, 1000, 2000], ["10", "100", "1,000", "2,000"])
ax.minorticks_off()
ax.set_ylim(10, 3000)
ax.set_xlim(12000, 36000)
ax.set_xticks([15000, 25000, 35000], ["15,000", "25,000", "35,000"])
ax.set_xlabel("genes DESeq2 tested")
ax.set_ylabel("genes")
ax.set_title("nf-core list sizes")
ax.legend(loc="upper left", frameon=False, fontsize=5, handletextpad=0.2, borderaxespad=0.1)
for a, l in zip(axs, "abc"):
    pf.panel_label(a, l)
fig.suptitle("each mark: one list. filled: beat random genes. open: did not. thick bar: all lists together. dotted line: 50 % target. "
             "text: genes reached, lists beating random", fontsize=5)
pf.save(fig, out_stem)
print("wrote", out_stem)
