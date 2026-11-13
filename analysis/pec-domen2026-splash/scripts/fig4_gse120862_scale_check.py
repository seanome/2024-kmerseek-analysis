#!/usr/bin/env python3
"""Why the GSE120862 score from reads goes the opposite way to the printed one (notebook 361).

Usage: fig4_gse120862_scale_check.py <geo_dir> <analysis_dir> <out_dir>

geo_dir holds GSE120862_FPKM.txt.gz and GSE120862_DESeq_normalized_counts.txt.gz, the two gene tables
the authors of the series put on GEO. analysis_dir is analysis/pec-domen2026-splash.

The test, made after the result was seen (so it explains, it does not confirm): the same score is
computed from several versions of the expression values that differ only in how each sample is
scaled. If the direction of the difference between untrained and trained samples turns with the
scaling, the printed AUC rests on the scaling, not on the listed genes.

Writes gse120862_scale_check.tsv (one row per version), gse120862_sample_totals.tsv (one row per
sample), gse120862_gene_changes.tsv.gz (one row per gene) and gse120862_scale_check.{png,pdf,svg}.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "notebooks"))
import fig4_score_from_reads as f  # noqa: E402
import pubfig as pf  # noqa: E402

geo, analysis, out = (Path(a) for a in sys.argv[1:4])
out.mkdir(parents=True, exist_ok=True)
clusters = {
    c: [g.strip().upper() for g in open(analysis / f"design/exercise_score/cluster_{c}.genes.txt") if g.strip()]
    for c in f.WEIGHTS
}
design = pd.read_csv(analysis / "design/human/samples.tsv", sep="\t")
design = design[design.series == "GSE120862"].set_index("run")
first = list(design.title[design.training == "untrained"])
second = list(design.title[design.training == "trained"])
samples = first + second
assert (len(first), len(second)) == (42, 42)


def score(expr):
    """The Fig. 4C-D score; expr is log2 values, one row per gene name, one column per sample."""
    values = f.gene_values(expr)
    found = {c: [g for g in genes if g in values.index] for c, genes in clusters.items()}
    s = sum(f.WEIGHTS[c] * values.loc[found[c]].sum() for c in f.WEIGHTS)
    return f.auc_of(s, first, second), f.auc_of(values.sum(), first, second), len(set().union(*found.values()))


def by_name(t):
    """One row per gene name: the values of rows with the same name are added."""
    t.index = t.index.str.upper()
    return t.groupby(level=0).sum()[samples]


fpkm_rows = pd.read_csv(geo / "GSE120862_FPKM.txt.gz", sep="\t").drop(columns="gene_id").set_index("gene_name")
fpkm = by_name(fpkm_rows)
deseq = by_name(pd.read_csv(geo / "GSE120862_DESeq_normalized_counts.txt.gz", sep="\t", index_col=0))
reads = pd.read_csv(
    analysis / "exports/fig4_score_from_reads/counts/GSE120862/share_1/gene.counts.tsv.gz", sep="\t", index_col=0
).rename(columns=design.title)[samples]
shared = fpkm.index.intersection(reads.index)  # genes with a read, which are protein-coding genes
total = fpkm.loc[shared].sum()  # each sample's summed FPKM over those genes
cpm = reads / reads.sum() * 1e6
# DESeq's scaling ("median of ratios"): a sample's factor is the median, over genes with a read in
# every sample, of its count divided by the gene's geometric mean over samples
every = reads[(reads > 0).all(axis=1)]
factor = np.exp(np.log(every).sub(np.log(every).mean(axis=1), axis=0).median())

versions = [
    ("authors' FPKM table, as on GEO", "authors' table", np.log2(fpkm + 1)),
    ("authors' FPKM table, genes with a read only", "authors' table", np.log2(fpkm.loc[shared] + 1)),
    (
        "authors' FPKM table, every sample scaled to the same total",
        "authors' table",
        np.log2(fpkm.loc[shared] / total * 1e6 + 1),
    ),
    ("authors' DESeq-normalised counts, as on GEO", "authors' table", np.log2(deseq + 1)),
    ("reads, no genome, every sample scaled to the same total (CPM)", "reads", np.log2(cpm + 1)),
    ("reads, no genome, scaled as DESeq does (median of ratios)", "reads", np.log2(reads / factor + 1)),
    (
        "reads, no genome, every sample given its total from the authors' FPKM table",
        "reads",
        np.log2(cpm * total / total.median() + 1),
    ),
]
rows = []
for name, source, expr in versions:
    auc, auc_all, n = score(expr)
    rows.append(
        dict(
            version=name,
            source=source,
            auc_listed_genes=auc,
            auc_all_genes_weight_1=auc_all,
            listed_genes_found=n,
            genes=len(expr),
        )
    )
    print(f"{auc:.3f}  all genes {auc_all:.3f}  listed {n}  {name}")
res = pd.DataFrame(rows)
res.to_csv(out / "gse120862_scale_check.tsv", sep="\t", index=False)

mito = fpkm.index.str.startswith("MT-")
tot = pd.DataFrame(
    {
        "sample": samples,
        "group": ["untrained"] * 42 + ["trained"] * 42,
        "fpkm_total_all_genes": fpkm.sum().to_numpy(),
        "fpkm_total_genes_with_a_read": total.to_numpy(),
        "fpkm_share_in_mitochondrial_genes": (fpkm[mito].sum() / fpkm.sum()).to_numpy(),
        "deseq_total": deseq.sum().to_numpy(),
        "reads_placed_in_a_gene": reads.sum().to_numpy(),
        "reads_total_after_deseq_scaling": (reads / factor).sum().to_numpy(),
    }
)
tot.to_csv(out / "gse120862_sample_totals.tsv", sep="\t", index=False)
g = tot.groupby("group", sort=False)
print(g.median(numeric_only=True).T.round(4))
print("AUC of the FPKM total itself, trained above untrained:", round(f.auc_of(total, first, second), 3))

# gene by gene: mean log2 value in trained minus untrained samples, authors' table against reads
la, lr = np.log2(fpkm.loc[shared] + 1), np.log2(cpm.loc[shared] + 1)
change = pd.DataFrame(
    {
        "authors_fpkm_trained_minus_untrained": la[second].mean(axis=1) - la[first].mean(axis=1),
        "reads_cpm_trained_minus_untrained": lr[second].mean(axis=1) - lr[first].mean(axis=1),
    }
)
change.to_csv(out / "gse120862_gene_changes.tsv.gz", sep="\t")
print(len(change), "genes; rank correlation of the change", round(change.corr(method="spearman").iloc[0, 1], 3))
print(change.median().round(3))

C = pf.OKABE_ITO
FIRST, SECOND = C["blue"], C["vermillion"]  # untrained, trained: as in fig4_score_from_reads
pf.use_style()
fig, (a, b) = pf.figure(pf.TWO_COLUMN_MM, 70, ncols=2, layout="constrained", gridspec_kw=dict(width_ratios=[1, 1.5]))
rng = np.random.default_rng(0)  # only for the sideways spread of the dots
for y, (grp, colour) in enumerate((("untrained", FIRST), ("trained", SECOND))):
    v = tot.fpkm_total_genes_with_a_read[tot.group == grp].to_numpy() / 1e3
    a.scatter(v, y + rng.uniform(-0.22, 0.22, len(v)), s=7, color=colour, linewidths=0, alpha=0.8)
    a.plot([np.median(v)] * 2, [y - 0.36, y + 0.36], color="black", lw=1)
a.set_yticks([0, 1], ["untrained (n = 42)", "trained (n = 42)"])
a.set_ylim(1.6, -0.6)
a.set_xlabel("Total FPKM of a sample in the authors' table\n(thousands; genes with a read; tick, median)")

order = res.iloc[::-1].reset_index(drop=True)
for y, r in order.iterrows():
    reads_row = r.source == "reads"
    b.scatter(
        r.auc_listed_genes,
        y,
        s=22,
        marker="o" if reads_row else "s",
        color=C["bluish_green"] if reads_row else C["reddish_purple"],
        zorder=3,
        label="score from reads, no genome" if reads_row else "score from the authors' table",
    )
    b.text(r.auc_listed_genes + 0.03, y, f"{r.auc_listed_genes:.2f}", va="center", fontsize=6)
b.axvline(0.08, color="black", lw=0.8, ls="--", label="printed AUC, 0.08")
b.axvline(0.5, color=pf.GREY, lw=0.5, ls=":", label="0.5, no difference")
b.set_yticks(
    range(len(order)), [v.replace(", every", ",\nevery").replace(", genes", ",\ngenes") for v in order.version]
)
b.set_ylim(-0.6, len(order) - 0.4)
b.set_xlim(0, 1)
b.set_xlabel("AUC, trained above untrained")
h, l = b.get_legend_handles_labels()
keep = dict(zip(l, h))
fig.legend(list(keep.values()), list(keep), loc="outside upper center", ncol=4, frameon=False)
pf.panel_label(a, "a", dx_pt=-60)
pf.panel_label(b, "b", dx_pt=-150)
pf.save(fig, str(out / "gse120862_scale_check"))
