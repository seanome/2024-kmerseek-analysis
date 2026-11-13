#!/usr/bin/env python3
"""KaMRaT contigs against the reads holding them, on the three small contrasts (notebook 361).

Usage: plot_kamrat_reads.py <reads_dir> <out_stem>

reads_dir holds recovery_real.tsv, recovery_swaps.tsv, contigs_swap_recovery.tsv, splash_recovery.tsv
and inputs.tsv.

a. List genes reached, of the six lists of these contrasts, going the same way as the list.
b. KaMRaT + DESeq2 per contrast, real labels against each swap.
c. All genes reached, in or out of the lists.
Writes <out_stem>.tsv with the totals and prints it. A swap's average is taken within each contrast
and then summed, so that the contrast with one swap counts as much as those with three.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "notebooks"))
import pubfig as pf  # noqa: E402

reads_dir, out_stem = Path(sys.argv[1]), sys.argv[2]
C = pf.OKABE_ITO
CONTRASTS = {"young_13_49d": "young 13-49 d", "young_132d": "young 132 d", "old_41_93d": "old 41-93 d"}
ROWS = {"splash": "SPLASH", "kamrat_ttest_p05": "KaMRaT t-test, contigs", "kamrat_ttest_p05_reads": "KaMRaT t-test, reads",
        "kamrat_deseq2_p05": "KaMRaT DESeq2, contigs", "kamrat_deseq2_p05_reads": "KaMRaT DESeq2, reads"}
real = pd.read_csv(reads_dir / "recovery_real.tsv", sep="\t")
real = pd.concat([real, pd.read_csv(reads_dir / "splash_recovery.tsv", sep="\t")])
swap_reads = pd.read_csv(reads_dir / "recovery_swaps.tsv", sep="\t")
swap_contigs = pd.read_csv(reads_dir / "contigs_swap_recovery.tsv", sep="\t").query("contrast in @CONTRASTS and run != 'real'")
inputs = pd.read_csv(reads_dir / "inputs.tsv", sep="\t").query("run == 'real'")
real["chance"] = real.null_median * real.list_genes_with_protein


def swap_mean(d):
    return d.groupby(["contrast", "run"]).reached.sum().groupby("contrast").mean().sum()


t = real.groupby("channel").agg(list_genes=("list_genes_with_protein", "sum"), reached=("reached", "sum"),
                                either_way=("reached_either_way", "sum"), chance=("chance", "sum"),
                                lists_beat_random=("share_null_at_or_above", lambda s: int((s <= 0.05).sum()))).loc[list(ROWS)]
t["swap_mean"] = float("nan")
t.loc["kamrat_deseq2_p05", "swap_mean"] = swap_mean(swap_contigs)
t.loc["kamrat_deseq2_p05_reads", "swap_mean"] = swap_mean(swap_reads)
pool = {"kamrat_ttest_p05": inputs.query("channel == 'kamrat_ttest_p05'").genes_reached_contigs.sum(),
        "kamrat_ttest_p05_reads": inputs.query("channel == 'kamrat_ttest_p05'").genes_reached_reads.sum(),
        "kamrat_deseq2_p05": inputs.query("channel == 'kamrat_deseq2_p05'").genes_reached_contigs.sum(),
        "kamrat_deseq2_p05_reads": inputs.query("channel == 'kamrat_deseq2_p05'").genes_reached_reads.sum()}
t["all_genes_reached"] = pd.Series(pool)
t.loc["splash", "either_way"] = float("nan")  # SPLASH has no direction
t = t.rename_axis("channel").reset_index()
t.to_csv(f"{out_stem}.tsv", sep="\t", index=False)
print(t.to_string(index=False))

pf.use_style()
fig, axs = pf.figure(pf.TWO_COLUMN_MM, 62, ncols=3, layout="constrained", gridspec_kw=dict(width_ratios=[1.15, 1.15, 0.7]))
KaMRaT, SPLASH, GREY = C["reddish_purple"], C["blue"], "0.45"
ax = axs[0]
ys = list(range(len(ROWS)))[::-1]
for y, r in zip(ys, t.itertuples()):
    col = SPLASH if r.channel == "splash" else KaMRaT
    ax.axhline(y, color="0.9", linewidth=0.5, zorder=0)
    ax.plot(r.reached, y, "o", color=col, markersize=4, zorder=3)
    ax.plot(r.chance, y, "x", color=GREY, markersize=3.5, markeredgewidth=0.7, zorder=2)
    if r.swap_mean == r.swap_mean:
        ax.plot(r.swap_mean, y, "o", markerfacecolor="white", markeredgecolor=col, markeredgewidth=0.7, markersize=4, zorder=2.5)
    ax.annotate(f"{r.reached}", (r.reached, y + 0.3), ha="center", fontsize=5, color=col)
ax.axvline(t.list_genes[0] / 2, color="0.45", linewidth=0.6, linestyle=":", zorder=1)
ax.set_yticks(ys, list(ROWS.values()))
ax.set_ylim(-0.6, len(ROWS) - 0.3)
ax.set_xlim(0, t.list_genes[0] * 0.75)
ax.set_xlabel(f"list genes reached, of {t.list_genes[0]}")
ax.set_title("all three contrasts")

ax = axs[1]
labels, y = [], 0
yt = []
for c, name in CONTRASTS.items():
    for ch, sw, kind in (("kamrat_deseq2_p05", swap_contigs, "contigs"), ("kamrat_deseq2_p05_reads", swap_reads, "reads")):
        ax.axhline(y, color="0.9", linewidth=0.5, zorder=0)
        ax.plot(sw[sw.contrast == c].groupby("run").reached.sum(), [y] * sw[sw.contrast == c].run.nunique(), "o", markerfacecolor="white",
                markeredgecolor=KaMRaT, markeredgewidth=0.7, markersize=4, zorder=2)
        ax.plot(real[(real.contrast == c) & (real.channel == ch)].reached.sum(), y, "o", color=KaMRaT, markersize=4, zorder=3)
        labels.append(f"{name}, {kind}")
        yt.append(y)
        y -= 1
    y -= 0.5
ax.set_yticks(yt, labels)
ax.set_xlim(0, None)
ax.set_xlabel("list genes reached")
ax.set_title("KaMRaT DESeq2: real labels against swaps")

ax = axs[2]
sub = t.dropna(subset=["all_genes_reached"]).reset_index(drop=True)
ysub = list(range(len(sub)))[::-1]
for y, r in zip(ysub, sub.itertuples()):
    ax.axhline(y, color="0.9", linewidth=0.5, zorder=0)
    ax.plot(r.all_genes_reached, y, "o", color=KaMRaT, markersize=4, zorder=3)
    ax.annotate(f"{r.all_genes_reached:.0f}", (r.all_genes_reached, y), xytext=(0, 4), textcoords="offset points", ha="left", fontsize=5, color=KaMRaT)
ax.set_yticks(ysub, ["t-test, contigs", "t-test, reads", "DESeq2, contigs", "DESeq2, reads"])
ax.set_ylim(-0.6, len(sub) - 0.3)
ax.set_xlim(0, sub.all_genes_reached.max() * 1.25)
ax.set_xlabel("all genes reached")
ax.set_title("in or out of the lists")
for a, l in zip(axs, "abc"):
    pf.panel_label(a, l)
handles = [pf.plt.Line2D([], [], marker="o", color=KaMRaT, linestyle="none", markersize=4, label="real labels"),
           pf.plt.Line2D([], [], marker="o", markerfacecolor="white", markeredgecolor=KaMRaT, linestyle="none", markersize=4, label="swapped labels (a: average, b: each swap)"),
           pf.plt.Line2D([], [], marker="x", color=GREY, linestyle="none", markersize=3.5, label="random genes of matched expression, median"),
           pf.plt.Line2D([], [], color="0.45", linewidth=0.6, linestyle=":", label="50 % of list genes")]
fig.legend(handles=handles, loc="outside upper center", ncols=4, frameon=False, fontsize=5)
pf.save(fig, out_stem)
print("wrote", out_stem)
