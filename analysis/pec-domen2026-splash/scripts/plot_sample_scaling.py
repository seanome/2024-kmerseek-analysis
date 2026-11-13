#!/usr/bin/env python3
"""Figure: the Fig. 4C-D score under two ways of scaling a sample (notebook 361).

Usage: plot_sample_scaling.py <analysis_dir> <out_dir>

Reads the count tables in exports/fig4_score_from_reads/counts (all reads) and the comparison tables
written by fig4_score_from_reads.py with and without size_factor, for the reads and for the authors'
tables, and exports/fig4_score_from_reads/gse120862_scale_check/gse120862_scale_check.tsv for the
authors' GSE120862 tables. Writes human_sample_totals.tsv, human_auc_by_scaling.tsv and
human_auc_by_scaling.{png,pdf,svg}. Plan: design/sample_scaling/README.md.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "notebooks"))
import fig4_score_from_reads as f  # noqa: E402
import pubfig as pf  # noqa: E402
from pec_analysis_utils import poscounts_size_factors  # noqa: E402

analysis, out = Path(sys.argv[1]), Path(sys.argv[2])
E = analysis / "exports"
design = pd.read_csv(analysis / "design/human/samples.tsv", sep="\t")
PANELS = {"GSE120862": "GSE120862 (panel C)", "GSE99963": "GSE99963 (panel D)", "GSE144304": "GSE144304 (panel D)"}

totals = []
for series, (column, name0, name1) in f.GROUPS.items():
    d = design[design.series == series].set_index("run")
    d = d[d[column].isin([name0, name1])]
    c = pd.read_csv(E / f"fig4_score_from_reads/counts/{series}/share_1/gene.counts.tsv.gz", sep="\t", index_col=0)
    factor = pd.Series(poscounts_size_factors(c.to_numpy()), index=c.columns)[d.index]
    total = c.sum()[d.index]
    rel = total / np.exp(np.log(total).mean()) / (factor / np.exp(np.log(factor).mean()))
    totals.append(
        pd.DataFrame(
            dict(
                series=series,
                run=d.index,
                group=d[column].to_numpy(),
                reads_in_a_gene=total.to_numpy(),
                size_factor=factor.to_numpy(),
                total_over_factor=rel.to_numpy(),
            )
        )
    )
totals = pd.concat(totals)
totals.to_csv(out / "human_sample_totals.tsv", sep="\t", index=False)


def all_reads(path, source, scaling):
    t = pd.read_csv(path, sep="\t")
    t = t[t.share == 1]
    return pd.DataFrame(
        dict(
            series=t.series,
            source=source,
            scaling=scaling,
            auc=t.auc_from_reads,
            printed_auc=t.printed_auc,
            rank_correlation_with_pr212=t.spearman_with_gene_table,
        )
    )


g = pd.read_csv(E / "fig4_score_from_reads/gse120862_scale_check/gse120862_scale_check.tsv", sep="\t").set_index(
    "version"
)
auc = pd.concat(
    [
        all_reads(E / "fig4_score_from_reads/fig4_reads_comparisons.tsv", "reads, no genome", "total"),
        all_reads(E / "sample_scaling/fig4_reads/fig4_reads_comparisons.tsv", "reads, no genome", "size factor"),
        all_reads(E / "fig4_score_from_reads/gene_table_check/fig4_reads_comparisons.tsv", "authors' table", "total"),
        all_reads(E / "sample_scaling/fig4_gene_tables/fig4_reads_comparisons.tsv", "authors' table", "size factor"),
        pd.DataFrame(
            dict(
                series="GSE120862",
                source="authors' table",
                scaling=["total", "size factor"],
                printed_auc=0.08,
                auc=[
                    g.auc_listed_genes["authors' FPKM table, every sample scaled to the same total"],
                    g.auc_listed_genes["authors' DESeq-normalised counts, as on GEO"],
                ],
            )
        ),
    ]
)
auc.to_csv(out / "human_auc_by_scaling.tsv", sep="\t", index=False)
print(auc.round(3).to_string())
print(totals.groupby(["series", "group"], sort=False).total_over_factor.median().round(3))

C = pf.OKABE_ITO
COLOUR = {"total": C["blue"], "size factor": C["vermillion"]}
MARK = {"authors' table": "s", "reads, no genome": "o"}
pf.use_style()
fig, axs = pf.figure(pf.TWO_COLUMN_MM, 85, nrows=2, ncols=3, layout="constrained")
rng = np.random.default_rng(0)  # only for the sideways spread of the dots
for j, (series, title) in enumerate(PANELS.items()):
    a = axs[0, j]
    t = totals[totals.series == series]
    groups = list(f.GROUPS[series][1:])  # first group, second group
    for y, grp in enumerate(groups):
        v = t.total_over_factor[t.group == grp].to_numpy()
        a.scatter(v, y + rng.uniform(-0.22, 0.22, len(v)), s=6, color=pf.GREY, linewidths=0)
        a.plot([np.median(v)] * 2, [y - 0.36, y + 0.36], color="black", lw=1)
    a.axvline(1, color=pf.GREY, lw=0.5, ls=":")
    a.set_yticks(range(2), [f"{grp} (n = {(t.group == grp).sum()})" for grp in groups])
    a.set_ylim(1.6, -0.6)
    a.set_title(title)
    a.set_xlabel("Total reads / size factor, per sample\n(1 = scalings agree; tick, median)")

    b = axs[1, j]
    s = auc[auc.series == series]
    for y, source in enumerate(MARK):
        for scaling in COLOUR:
            r = s[(s.source == source) & (s.scaling == scaling)].iloc[0]
            b.scatter(r.auc, y, s=22, marker=MARK[source], color=COLOUR[scaling], zorder=3)
            b.annotate(
                f"{r.auc:.2f}",
                (r.auc, y),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                fontsize=6,
                bbox=dict(fc="white", ec="none", pad=0.4),
                zorder=4,
            )
    printed = s.printed_auc.iloc[0]
    b.axvline(printed, color="black", lw=0.8, ls="--")
    b.axvline(0.5, color=pf.GREY, lw=0.5, ls=":")
    b.set_yticks(range(2), list(MARK))
    b.set_ylim(1.6, -0.7)
    b.set_xlim(0, 1)
    b.set_xlabel(f"AUC, second group above first\nprinted {printed:.2f}")

from matplotlib.lines import Line2D  # noqa: E402

handles = [
    Line2D([], [], marker="o", ls="", color=COLOUR["total"], ms=4, label="each sample divided by its total (CPM)"),
    Line2D(
        [], [], marker="o", ls="", color=COLOUR["size factor"], ms=4, label="each sample divided by its size factor"
    ),
    Line2D([], [], marker="s", ls="", color="black", ms=4, label="authors' table"),
    Line2D([], [], marker="o", ls="", color="black", ms=4, label="reads, no genome"),
    Line2D([], [], color="black", lw=0.8, ls="--", label="printed AUC"),
    Line2D([], [], color=pf.GREY, lw=0.5, ls=":", label="no difference"),
]
fig.legend(handles=handles, loc="outside upper center", ncol=3, frameon=False)
for ax, letter in zip(axs[:, 0], "ab"):
    pf.panel_label(ax, letter, dx_pt=-60)
pf.save(fig, str(out / "human_auc_by_scaling"))

# ---- Botryllus: the up/down rule under the two scalings, against nf-core's lists and the paper's lists
B = out / "botryllus"
if (B / "nfcore/recovery.tsv").exists():
    n = pd.read_csv(B / "nfcore/recovery.tsv", sep="\t")
    n = n[n.cutoff == "paper_cutoff"]
    # nfcore_recovery.py's two count routes were given sample_mean and size_factor_mean, in that order
    parts = [
        n[n.channel == ch].assign(scaling=sc, lists="nf-core's lists", genes=lambda d: d.list_genes)
        for ch, sc in (("gene_counts", "total"), ("gene_counts_mean", "size factor"))
    ]
    for rule, sc in (("sample_mean", "total"), ("size_factor_mean", "size factor")):
        p = pd.read_csv(B / f"paper_{rule}/recovery.tsv", sep="\t")
        parts.append(p.assign(scaling=sc, lists="the paper's lists", genes=p.list_genes_with_protein))
    bot = pd.concat(parts)[["lists", "scaling", "contrast", "list", "genes", "reached", "share_null_at_or_above"]]
    bot["beats_random_gene_sets"] = bot.share_null_at_or_above <= 0.05
    bot.to_csv(out / "botryllus_lists_by_scaling.tsv", sep="\t", index=False)
    tot = bot.groupby(["lists", "scaling"], sort=False).agg(
        genes=("genes", "sum"),
        reached=("reached", "sum"),
        lists_beating_random=("beats_random_gene_sets", "sum"),
        n_lists=("list", "size"),
    )
    print(tot)
    fig, axs = pf.figure(pf.TWO_COLUMN_MM, 55, ncols=2, layout="constrained", sharey=True)
    for ax, lists in zip(axs, ("nf-core's lists", "the paper's lists")):
        for y, sc in enumerate(COLOUR):
            d = bot[(bot.lists == lists) & (bot.scaling == sc) & (bot.genes > 0)]
            ax.scatter(
                100 * d.reached / d.genes,
                y + rng.uniform(-0.2, 0.2, len(d)),
                s=4 + 0.25 * d.genes,
                facecolor="none",
                edgecolor=COLOUR[sc],
                linewidths=0.6,
            )
            t = tot.loc[(lists, sc)]
            ax.plot([100 * t.reached / t.genes] * 2, [y - 0.36, y + 0.36], color=COLOUR[sc], lw=2)
            ax.annotate(
                f"{int(t.reached)} of {int(t.genes)} ({100 * t.reached / t.genes:.0f} %)",
                (100 * t.reached / t.genes, y - 0.42),
                ha="center",
                va="bottom",
                fontsize=6,
            )
        ax.set_title(f"Botryllus, {lists}")
        ax.set_xlim(0, 100)
        ax.set_ylim(1.6, -0.9)
        ax.set_yticks(range(2), ["each sample divided by\nits total (CPM)", "each sample divided by\nits size factor"])
        ax.set_xlabel("List genes given the list's direction from reads, no genome (%)")
    handles = [
        Line2D([], [], marker="o", ls="", mfc="none", color="black", ms=3, label="one list of 20 genes"),
        Line2D([], [], marker="o", ls="", mfc="none", color="black", ms=8, label="one list of 250 genes"),
        Line2D([], [], color="black", lw=2, label="all 12 lists together"),
    ]
    fig.legend(handles=handles, loc="outside upper center", ncol=3, frameon=False)
    pf.save(fig, str(out / "botryllus_lists_by_scaling"))
