#!/usr/bin/env python3
"""Which Botryllus genes pass in human and Botryllus reads, by alphabet and k (notebook 361).

Usage: plot_sra_alphabet_overlap.py <alphabet_reach dir> <hp24_27_reach dir> <pd_k10_reach dir> <out_stem>

Inputs are gene_passes.tsv and threshold.tsv from scripts/sra_alphabet_reach.py. The folders come
from two sketching methods: alphabet_reach used Swiss-Prot 7-mer frames at scaled 10; hp24_27_reach (HP)
and pd_k10_reach (protein, Dayhoff) used each species' own protein 10-mer frames at scaled 1. Protein k7,
Dayhoff k12, k13 and HP k27 were run both ways, which shows how much the method alone changes the count.

a. Per setting and species: paper-list genes passing (of 823) and random genes passing (of 848).
   Settings where even the shuffled proteins' 99th percentile is 1.0 cannot pass anything; they get
   a grey x instead of a count.
b. Human: how the passing list genes split across protein k7, Dayhoff k13 and HP k25 (all 108 runs,
   all made with each species' own frames at scaled 1), and the same split for random genes.
Writes <out_stem>.tsv (panel a) and <out_stem>.overlap.tsv (panel b), and prints both.
"""
import sys
from itertools import product
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "notebooks"))
import pubfig as pf  # noqa: E402

old_dir, new_dir, pd_dir, out_stem = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
C = pf.OKABE_ITO


def load(d, method):
    g = pl.read_csv(d / "gene_passes.tsv", separator="\t").with_columns(method=pl.lit(method))
    t = pl.read_csv(d / "threshold.tsv", separator="\t").with_columns(method=pl.lit(method))
    return g, t


g0, t0 = load(old_dir, "Swiss-Prot 7-mer frames, scaled 10")
g1, t1 = load(new_dir, "own 10-mer frames, scaled 1")
g2, t2 = load(pd_dir, "own 10-mer frames, scaled 1")
g1, t1 = pl.concat([g1, g2]), pl.concat([t1, t2])
g, t = pl.concat([g0, g1]), pl.concat([t0, t1])
summ = (g.filter(pl.col("set") != "shuffled").group_by("method", "alphabet", "k", "species", "set")
        .agg(genes=pl.len(), passed=pl.col("passed").sum())
        .pivot(on="set", index=["method", "alphabet", "k", "species"], values=["genes", "passed"])
        .join(t, on=["method", "alphabet", "k", "species"], how="left")
        .sort("species", "method", "alphabet", "k"))
summ.write_csv(f"{out_stem}.tsv", separator="\t")
with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=200):
    print(summ)

TRIO = [("protein", 7, g1), ("dayhoff", 13, g1), ("hp", 25, g1)]
over = []
for s in ("paper_list", "random_length_matched"):
    sets = [set(x.filter((pl.col("alphabet") == a) & (pl.col("k") == k) & (pl.col("species") == "human")
                         & (pl.col("set") == s) & pl.col("passed"))["gene"]) for a, k, x in TRIO]
    allg = set().union(*sets)
    for combo in product([True, False], repeat=3):
        if not any(combo):
            continue
        n = sum(all((x in sets[i]) == combo[i] for i in range(3)) for x in allg)
        over.append(dict(set=s, protein_k7=combo[0], dayhoff_k13=combo[1], hp_k25=combo[2], genes=n))
over = pl.DataFrame(over)
over.write_csv(f"{out_stem}.overlap.tsv", separator="\t")
print(over)

pf.use_style()
fig, axs = pf.figure(pf.TWO_COLUMN_MM, 95, ncols=3, layout="constrained", gridspec_kw=dict(width_ratios=[1, 1, 1.1]))
ORDER = [("protein", 6, 0), ("protein", 7, 1), ("protein", 7, 0), ("dayhoff", 12, 1), ("dayhoff", 12, 0), ("dayhoff", 13, 1),
         ("dayhoff", 13, 0), ("hp", 24, 1), ("hp", 25, 1), ("hp", 26, 1), ("hp", 27, 1), ("hp", 27, 0), ("hp", 30, 0)]
methods = [g0["method"][0], g1["method"][0]]
ys, labels, y = [], [], 0
for i, (a, k, m) in enumerate(ORDER):
    if i and ORDER[i - 1][0] != a:
        y += 0.6
    ys.append(y)
    labels.append(f"{'Dayhoff' if a == 'dayhoff' else a.replace('hp', 'HP')} k{k}{' *' if m == 0 else ''}")
    y += 1
ys = [max(ys) - v for v in ys]
for ax, sp in zip(axs[:2], ("human", "Botryllus")):
    for yy, (a, k, m) in zip(ys, ORDER):
        ax.axhline(yy, color="0.9", linewidth=0.5, zorder=0)
        r = summ.filter((pl.col("species") == sp) & (pl.col("alphabet") == a) & (pl.col("k") == k)
                        & (pl.col("method") == methods[m]))
        if r.is_empty():
            ax.annotate("no reads long enough", (0.01, yy), xycoords=("axes fraction", "data"), va="center", fontsize=5, color="0.4")
            continue
        r = r.row(0, named=True)
        if r["cut"] >= 1.0:
            ax.scatter([0.015], [yy], marker="x", color="0.5", s=8, linewidths=0.6, zorder=2)
            ax.annotate("shuffled proteins reach 100 %", (0.015, yy), va="center", fontsize=5, color="0.4",
                        xytext=(4, 0), textcoords="offset points")
            continue
        pl_, pr_ = r["passed_paper_list"] / r["genes_paper_list"], r["passed_random_length_matched"] / r["genes_random_length_matched"]
        ax.plot([pr_, pl_], [yy, yy], color="0.6", linewidth=0.8, zorder=1)
        ax.scatter([pr_], [yy], color="white", edgecolors=C["black"], s=12, linewidths=0.6, zorder=2, label="random genes")
        ax.scatter([pl_], [yy], color=C["black"], s=12, zorder=3, label="paper's list genes")
        ax.annotate(f"{r['passed_paper_list']} vs {r['passed_random_length_matched']}  ({r['runs']} runs)", (pl_, yy),
                    xytext=(4, 0), textcoords="offset points", va="center", fontsize=5)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0", "25", "50", "75", "100"])
    ax.set_xlabel("genes passing (%)")
    ax.set_title(f"{sp} reads")
    ax.set_ylim(min(ys) - 0.6, max(ys) + 0.6)
axs[0].set_yticks(ys, labels)
axs[1].set_yticks(ys, [])
h, lab = axs[0].get_legend_handles_labels()
keep = {l_: h_ for h_, l_ in zip(h, lab)}
axs[0].legend(keep.values(), keep.keys(), loc="lower left", bbox_to_anchor=(0, 1.08), ncol=2, frameon=False, fontsize=5)

ax = axs[2]
T, F = True, False
combos = [(T, T, T), (T, T, F), (T, F, T), (F, T, T), (T, F, F), (F, T, F), (F, F, T)]
names = ["protein k7", "Dayhoff k13", "HP k25"]
w = 0.38
for j, c in enumerate(combos):
    for s, off, fc in (("paper_list", -w / 2, C["black"]), ("random_length_matched", w / 2, "white")):
        n = over.filter((pl.col("set") == s) & (pl.col("protein_k7") == c[0]) & (pl.col("dayhoff_k13") == c[1])
                        & (pl.col("hp_k25") == c[2]))["genes"][0]
        ax.bar(j + off, n, width=w, color=fc, edgecolor=C["black"], linewidth=0.5)
        ax.annotate(str(n), (j + off, n), xytext=(0, 1), textcoords="offset points", ha="center", fontsize=5)
    for i in range(3):
        ax.scatter([j], [-0.07 - 0.06 * i], s=10, color=C["black"] if c[i] else "0.85", clip_on=False, zorder=3,
                   transform=ax.get_xaxis_transform())
for i, nm in enumerate(names):
    ax.annotate(nm, (-0.6, -0.07 - 0.06 * i), xycoords=ax.get_xaxis_transform(), ha="right", va="center", fontsize=5,
                annotation_clip=False)
ax.set_xticks([])
ax.set_xlim(-0.6, len(combos) - 0.4)
ax.set_ylim(0, None)
ax.set_ylabel("human: genes passing")
ax.set_title("human: which settings a gene passes in")
ax.legend([pf.plt.Rectangle((0, 0), 1, 1, fc=C["black"]), pf.plt.Rectangle((0, 0), 1, 1, fc="white", ec=C["black"])],
          ["paper's list genes", "random genes"], loc="upper right", frameon=False, fontsize=5)
for a_, l_ in zip(axs, "abc"):
    pf.panel_label(a_, l_)
fig.suptitle("A gene passes if a run holds more of its k-mers than 99 % of shuffled proteins do.  * = Swiss-Prot 7-mer "
             "frames, scaled 10; others: each species' own 10-mer frames, scaled 1", fontsize=5)
pf.save(fig, out_stem)
print("wrote", out_stem)
