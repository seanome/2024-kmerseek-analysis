#!/usr/bin/env python3
"""Why the same routes find most genes in some data sets and few in others (notebook 361).

Usage: plot_why_control_differs.py <exports_dir> <out_stem>

a. Share of genes found by three routes in the simulated control (400 planted genes), the human
   series with swaps (12 study lists, GSE99963 and GSE144304) and Botryllus (the paper's 12 lists).
   Found: a kept sequence lies in the gene (control), or reaches its protein going the list's way
   (human and Botryllus; SPLASH has no direction, so any way).
b. Reads per sample, one dot per sample.
c. Reads per sample of each gene: box 25-75 %, tick the median. Human and Botryllus: baseMean of the
   list genes. Control: all its genes, read pairs per sample from pec-tools gene-counts.
Writes <out_stem>.tsv with the numbers in a, and prints them.
"""
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "notebooks"))
import pubfig as pf  # noqa: E402

ex, out_stem = Path(sys.argv[1]), sys.argv[2]
C = pf.OKABE_ITO
ROUTES = {"kamrat_ttest_p05": ("KaMRaT, t-test, p < 0.05", C["reddish_purple"]),
          "kamrat_deseq2_p05": ("KaMRaT, DESeq2, p < 0.05", C["reddish_purple"]),
          "splash": ("SPLASH, corrected p ≤ 0.05", C["blue"])}
DATA = ["simulated control", "human", "Botryllus"]
PLANTED = ["abundance_up", "abundance_down", "isoform_switch"]

rows = []
ch = json.load(open(ex / "control/control_truth_summary_p05.json"))["channels"]
for r in ROUTES:
    rows.append(dict(data="simulated control", route=r, found=sum(ch[r]["genes_found"][c] for c in PLANTED),
                     genes=sum(ch[r]["genes_in_class"][c] for c in PLANTED)))
h = pl.read_csv(ex / "human/human_compare.tsv", separator="\t").filter(~pl.col("contrast").str.starts_with("gse120862"))
for r in ROUTES:
    x = h.filter(pl.col("route") == ("splash_compactors" if r == "splash" else r))
    assert x.height == 12
    rows.append(dict(data="human", route=r, found=x["reached"].sum(), genes=x["list_genes"].sum()))
b = pl.read_csv(ex / "paper_recovery/paper_recovery.tsv", separator="\t")
for r in ROUTES:
    x = b.filter(pl.col("channel") == r)
    assert x.height == 12
    rows.append(dict(data="Botryllus", route=r, found=x["reached"].sum(), genes=x["list_genes_with_protein"].sum()))
t = pl.DataFrame(rows).with_columns((pl.col("found") / pl.col("genes")).round(3).alias("share"))
t.write_csv(f"{out_stem}.tsv", separator="\t")
print(t)

d = pl.read_csv(ex / "cheap_route/reads_per_sample_human_control.tsv", separator="\t")
bot = pl.read_csv(ex / "read_depth/read_depth_raw.tsv", separator="\t", has_header=False, new_columns=["c", "sample", "g", "reads"])
human_runs = {s.strip() for f in Path(ex.parent / "design/human").glob("contrast_gse*.cjs") if "gse120862" not in f.name
              for s in pl.read_csv(f, has_header=False)[:, 0].to_list()}
reads = {"simulated control": d.filter(pl.col("dataset") == "control")["reads"].to_numpy(),
         "human": d.filter((pl.col("dataset") == "human") & pl.col("sample").is_in(list(human_runs)))["reads"].to_numpy(),
         "Botryllus": bot["reads"].to_numpy()}
pub = pl.read_csv(ex / "human/published_genes.tsv", separator="\t").filter(~pl.col("contrast").str.starts_with("gse120862"))
bm_h = pub.join(pl.read_csv(ex / "human/gene_basemean.tsv.gz", separator="\t"), on=["contrast", "gene"])["baseMean"].to_numpy()
bm_b = pl.read_csv(ex / "cheap_route/list_gene_basemean_botryllus.tsv", separator="\t")["baseMean"].to_numpy()
for k, v in reads.items():
    print(k, "samples", len(v), "reads min", v.min(), "median", np.median(v), "max", v.max())
print("list gene baseMean median: human", np.median(bm_h), "of", len(bm_h), "; Botryllus", np.median(bm_b), "of", len(bm_b))

pf.use_style()
fig = pf.plt.figure(figsize=(pf.TWO_COLUMN_MM / 25.4, 95 / 25.4), layout="constrained")
top, bottom = fig.subfigures(2, 1, height_ratios=[1.15, 1])
axs = top.subplots(1, 3, sharey=True, sharex=True)
yr = {r: i for i, r in enumerate(ROUTES)}
for ax, data in zip(axs, DATA):
    for x in t.filter(pl.col("data") == data).iter_rows(named=True):
        lab, col = ROUTES[x["route"]]
        ax.barh(yr[x["route"]], x["share"], color=col, height=0.65)
        ax.text(1.03, yr[x["route"]], f"{x['share']:.0%}  ({x['found']:,} of {x['genes']:,})", va="center", fontsize=5.5)
    ax.axvline(0.5, color="black", linewidth=0.6, linestyle="--")
    ax.set_title(data)
    ax.set_xlim(0, 1.75)
    ax.set_xticks([0, 0.5, 1])
    ax.set_xlabel("share of genes found", x=0.29)
top.suptitle("dashed line: the 50 % target", fontsize=6)
axs[0].set_yticks(list(yr.values()), [ROUTES[r][0] for r in ROUTES])
axs[0].set_ylim(2.6, -0.6)
pf.panel_label(axs[0], "a")
ax1, ax2 = bottom.subplots(1, 2, sharey=True)
rng = np.random.default_rng(361)
yd = {k: i for i, k in enumerate(DATA)}
for k, v in reads.items():
    ax1.plot(v, yd[k] + rng.uniform(-0.2, 0.2, len(v)), "o", color="black", markersize=1.5, markeredgewidth=0, alpha=0.6)
ax1.set_xscale("log")
ax1.set_xlabel("reads per sample (one dot per sample)")
ax1.set_yticks(list(yd.values()), DATA)
ax1.set_ylim(2.6, -0.6)
pf.panel_label(ax1, "b")
cc = pl.read_csv(ex / "cheap_route/control/control.counts.tsv", separator="\t")
bm_c = cc.select(pl.mean_horizontal(cc.columns[1:]).alias("m"))["m"].to_numpy()
print("control reads per gene, median", np.median(bm_c))
for k, v in {"simulated control": bm_c, "human": bm_h, "Botryllus": bm_b}.items():
    q1, m, q3 = np.quantile(v, [0.25, 0.5, 0.75])
    ax2.plot([q1, q3], [yd[k]] * 2, color=pf.GREY, linewidth=5, solid_capstyle="butt")
    ax2.plot([m], [yd[k]], marker="|", color="black", markersize=8, markeredgewidth=1)
    ax2.text(m, yd[k] - 0.32, f"median {m:,.1f}", ha="center", fontsize=5.5)
ax2.set_xscale("log")
ax2.set_xlim(1, 3000)
ax2.set_xlabel("reads per sample for a gene (box: middle 50 %, tick: median)")
pf.panel_label(ax2, "c")
pf.save(fig, out_stem)
print("wrote", out_stem)
