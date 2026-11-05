#!/usr/bin/env python3
"""Genes reached against resources, Botryllus, every route (notebook 361).

Usage: plot_gain_vs_cost.py <resources.tsv> <routes_vs_paper_and_nfcore.tsv> <thin_dir> <full_old.time> <full_new.time> <out_stem>

Rows: the paper's lists, nf-core's lists. Columns: CPU hours, wall time, peak memory, each axis reversed,
so up and to the right is better. y: list genes reached in the list's direction (SPLASH has no
direction), summed over the 12 lists. Dotted lines: 50 % of genes, and the 1/1000 cost target.
Routes other than gene-counts add up their steps in all six contrasts, from resources.tsv; peak memory is
the largest step. gene-counts: one `/usr/bin/time -v` run over all 47 samples, mean-of-samples rule; the
line joins the same rule at more reads. <thin_dir> holds {old,new}.{10,100,1000}/gene_counts.time and
{old,new}.{10,100,1000}.reached.tsv.
Writes <out_stem>.tsv with every number drawn, and prints it.
"""
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "notebooks"))
import pubfig as pf  # noqa: E402

res_tsv, share_tsv, thin, full_old, full_new, out_stem = sys.argv[1:]
thin = Path(thin)
C = pf.OKABE_ITO
ROUTES = {"splash": ("SPLASH", C["blue"], "o"), "anchor_nominal": ("anchor counts, p < 0.05", C["vermillion"], "s"),
          "kamrat_ttest_p05": ("KaMRaT, t-test, p < 0.05", C["reddish_purple"], "o"),
          "kamrat_deseq2_p05": ("KaMRaT, DESeq2, p < 0.05", C["reddish_purple"], "s"),
          "dekupl_limma_p05": ("DE-kupl, limma, p < 0.05", C["bluish_green"], "o"),
          "dekupl_deseq2_p05": ("DE-kupl, DESeq2, p < 0.05", C["orange"], "s")}
TARGET = dict(cpu_hours=44 / 3600, wall_minutes=11 / 60, peak_gb=0.084)


def time_file(p):
    t = open(p).read()
    wall = re.search(r"Elapsed.*: (\S+)\n", t).group(1).split(":")
    wall = sum(float(x) * 60 ** i for i, x in enumerate(reversed(wall)))
    cpu = sum(float(x) for x in re.findall(r"(?:User|System) time \(seconds\): (\S+)", t))
    return dict(cpu_hours=cpu / 3600, wall_minutes=wall / 60, peak_gb=int(re.search(r"Maximum resident.*: (\d+)", t).group(1)) / 1e6)


res = pd.read_csv(res_tsv, sep="\t")
cost = res.groupby("route").agg(cpu_hours=("cpu_h", "sum"), wall_h=("wall_h", "sum"), peak_gb=("peak_gb", "max"))
cost["wall_minutes"] = cost.wall_h * 60
share = pd.read_csv(share_tsv, sep="\t")
rows = []
for ref in ("paper", "nf-core"):
    s = share[share.reference == ref].set_index("channel")
    for ch in ROUTES:
        rows.append(dict(reference=ref, route=ch, reads="all", genes=s.genes[ch], reached=s.reached[ch],
                         **cost.loc[ch, ["cpu_hours", "wall_minutes", "peak_gb"]].to_dict()))
    tag, full = ("old", full_old) if ref == "paper" else ("new", full_new)
    rows.append(dict(reference=ref, route="gene_counts_mean", reads="all", genes=s.genes["gene_counts_mean"],
                     reached=s.reached["gene_counts_mean"], **time_file(full)))
    for n in (10, 100, 1000):
        g = pd.read_csv(thin / f"{tag}.{n}.reached.tsv", sep="\t").iloc[0]
        rows.append(dict(reference=ref, route="gene_counts_mean", reads=f"1/{n}", genes=g.genes, reached=g.reached,
                         lists_beat_random=g.lists_beat_random, **time_file(thin / f"{tag}.{n}/gene_counts.time")))
t = pd.DataFrame(rows)
t["pct"] = 100 * t.reached / t.genes
t.to_csv(f"{out_stem}.tsv", sep="\t", index=False)
print(t.round(3).to_string(index=False))

COSTS = [("cpu_hours", "CPU hours", [0.01, 0.1, 1, 10, 100], ["0.01", "0.1", "1", "10", "100"]),
         ("wall_minutes", "wall time (minutes)", [0.1, 1, 10, 100, 1000], ["0.1", "1", "10", "100", "1,000"]),
         ("peak_gb", "peak memory (GB)", [0.05, 0.5, 5, 50], ["0.05", "0.5", "5", "50"])]
pf.use_style()
fig, axs = pf.figure(pf.TWO_COLUMN_MM, 120, nrows=2, ncols=3, sharex="col", sharey="row", layout="constrained")
for i, ref in enumerate(("paper", "nf-core")):
    d = t[t.reference == ref]
    g = d[d.route == "gene_counts_mean"]
    for j, (cc, lab, ticks, labels) in enumerate(COSTS):
        ax = axs[i, j]
        ax.axhline(50, color="0.45", linewidth=0.6, linestyle=":", zorder=1)
        ax.axvline(TARGET[cc], color="0.45", linewidth=0.6, linestyle=":", zorder=1)
        for ch, (name, col, mk) in ROUTES.items():
            r = d[d.route == ch].iloc[0]
            ax.plot(r[cc], r.pct, marker=mk, markersize=4, color=col, linestyle="none", label=name if (i, j) == (0, 0) else None, zorder=3)
        ax.plot(g[cc], g.pct, color="black", marker="D", markersize=3.5, linewidth=0.8, label="gene-counts, mean of samples" if (i, j) == (0, 0) else None, zorder=3)
        for r in g.itertuples():
            ax.annotate(f"{r.reads} reads" if r.reads == "all" else r.reads, (getattr(r, cc), r.pct), xytext=(3, 4), textcoords="offset points", fontsize=5)
        ax.set_ylim(0, 100)
        if i == 1:
            ax.set_xscale("log")
            ax.set_xticks(ticks, labels)
            ax.minorticks_off()
            ax.invert_xaxis()
            ax.set_xlabel(f"{lab}\n← more    less →")
    axs[i, 0].set_ylabel("list genes reached,\ngoing the same way (%)")
    axs[i, 0].set_title(f"{'the paper' if ref == 'paper' else 'nf-core'}'s lists", loc="left")
for a, l in zip(axs.flat, "abcdef"):
    pf.panel_label(a, l)
fig.legend(*axs[0, 0].get_legend_handles_labels(), loc="outside upper center", ncols=4, frameon=False,
           title="up and to the right is better.  dotted lines: 50 % of genes, and 1/1000 of the SPLASH route's cost")
pf.save(fig, out_stem)
print("wrote", out_stem)
