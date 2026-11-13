#!/usr/bin/env python3
"""Each human study's own gene-level result, per contrast (notebook 361, added 2026-11-03).

Usage: human_published_genes.py <pub_dir> <design_dir> <gencode.v50.pc_translations.fa.gz> <out_dir> [n_cpus]

pub_dir holds the files below, downloaded on 2026-11-03 (URLs and sha256 in
design/human/README.md):
  laker_S1b.xlsx                          Laker et al. 2017, Sci Rep 7:15134, Table S1
                                          (edgeR, ~ time + subject per arm; the paper's cutoff FDR < 0.1)
  GSE99963_GeneCoverage.tsv.gz            the authors' gene counts (used only for gene baseMean)
  GSE144304_JoinedCounts.txt.gz           the authors' gene counts (htseq-count)
  GSE120862_DESeq_normalized_counts.txt.gz the authors' DESeq-normalised counts

Laker 2017 published its gene lists. For GSE144304 (de Jong et al. 2023, GeroScience 45:2367;
van der Hoek et al. 2020) and GSE120862 (Popov et al. 2019, Am J Physiol Endocrinol Metab
316:E605) no table of differentially expressed genes was found, so a list is rebuilt from the
authors' own GEO count table with PyDESeq2 and labelled "rebuilt", not "published":
  GSE144304: ~ condition per sex, uncorrected p < 0.01 (de Jong 2023's cutoff for old vs young;
             the same cutoff is used for frail vs fit).
  GSE120862: ~ subject + condition, BH-corrected p < 0.05 (the paper's cutoff could not be read;
             the journal page refused access). The GEO table is already normalised, so its values
             are rounded and DESeq2's size factors come out close to 1.

Writes <out_dir>/published_genes.tsv (series, contrast, source, direction, gene) and
<out_dir>/gene_basemean.tsv (series, contrast, gene, baseMean: the pool for the random-gene null).
Genes are named by symbol; Ensembl IDs are mapped to symbols through the GENCODE v50
translations, so only protein-coding genes with a GENCODE protein are kept.
"""
import gzip
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pydeseq2.dds import DeseqDataSet
from pydeseq2.default_inference import DefaultInference
from pydeseq2.ds import DeseqStats

pub, design_dir, gencode, out = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4])
n_cpus = int(sys.argv[5]) if len(sys.argv) > 5 else 4
out.mkdir(parents=True, exist_ok=True)
inf = DefaultInference(n_cpus=n_cpus)

ens2sym = {}
with gzip.open(gencode, "rt") as f:
    for line in f:
        if line.startswith(">"):
            p = line[1:].strip().split("|")
            ens2sym[p[2].split(".")[0]] = p[6]
coding = set(ens2sym.values())
samples = pd.read_csv(design_dir / "samples.tsv", sep="\t", dtype=str)
contrasts = pd.read_csv(design_dir / "contrasts.tsv", sep="\t")
rows, pools = [], []


def design_of(c):
    return pd.read_csv(design_dir / f"contrast_{c}.design.tsv", sep="\t", dtype=str)


def deseq(counts, meta, formula):
    """counts: genes x samples (ints); meta: index samples. Returns results_df (PEC vs control)."""
    keep = (counts > 0).sum(1) >= int(np.ceil(counts.shape[1] / 2))
    dds = DeseqDataSet(counts=counts.loc[keep].T, metadata=meta, design=formula, refit_cooks=True, inference=inf)
    dds.deseq2()
    st = DeseqStats(dds, contrast=["condition", "PEC", "control"], inference=inf)
    st.summary()
    return st.results_df


def add_list(series, c, source, genes_up, genes_down):
    for d, gs in (("up", genes_up), ("down", genes_down)):
        for g in sorted(set(gs) & coding):
            rows.append((series, c, source, d, g))


def add_pool(series, c, base):
    """base: Series gene symbol -> baseMean (mean normalised count)."""
    b = base[base.index.isin(coding)]
    b = b.groupby(level=0).sum()
    for g, v in b.items():
        pools.append((series, c, g, float(v)))


# ---- GSE99963: Laker 2017 Table S1, FDR < 0.1
xl = pd.ExcelFile(pub / "laker_S1b.xlsx")
cov = pd.read_csv(pub / "GSE99963_GeneCoverage.tsv.gz", sep="\t", index_col=0)
s99 = samples[samples.series == "GSE99963"].set_index("title")
for c, sheet in (("gse99963_sed_post_vs_pre", "HFD_Sed_RNAseq"), ("gse99963_ex_post_vs_pre", "HFD_Ex_RNAseq")):
    t = xl.parse(sheet)
    t = t[t["FDR"] < 0.1]
    print(c, "Table S1 rows at FDR < 0.1:", len(t), "up", int((t.logFC > 0).sum()), "down", int((t.logFC < 0).sum()))
    add_list("GSE99963", c, "published (Laker 2017 Table S1, FDR < 0.1)",
             t.loc[t.logFC > 0, "Gene.Name"].dropna(), t.loc[t.logFC < 0, "Gene.Name"].dropna())
    runs = set(design_of(c)["sample"])
    cols = [ti for ti in cov.columns if ti in s99.index and s99.loc[ti, "run"] in runs]
    x = cov[cols]
    lib = x.sum(0)
    base = (x / lib * lib.mean()).mean(1)
    base.index = [ens2sym.get(i.split(".")[0], "") for i in base.index]
    add_pool("GSE99963", c, base[base.index != ""])

# ---- GSE99963, Fig. 4 groups: no published list pools the arms, so rebuilt from the authors' counts,
# ~ subject + condition, BH p < 0.05 (the GSE120862 rule), added 2026-11-04
c = "gse99963_fig4_post_vs_pre"
dsg = design_of(c).merge(s99.reset_index()[["run", "title"]], left_on="sample", right_on="run")
x = cov[dsg["title"].tolist()].round().astype(int)
x.columns = dsg["sample"].tolist()
x = x[x.index.map(lambda i: ens2sym.get(i.split(".")[0], "") != "")]
x.index = [ens2sym[i.split(".")[0]] for i in x.index]
x = x.groupby(level=0).sum()
r = deseq(x, dsg.set_index("sample")[["subject", "condition"]], "~ subject + condition")
sig = r[r.padj < 0.05]
print(c, "rebuilt padj < 0.05:", len(sig), "up", int((sig.log2FoldChange > 0).sum()), "down", int((sig.log2FoldChange < 0).sum()))
add_list("GSE99963", c, "rebuilt (PyDESeq2 on GEO counts, ~ subject + condition, padj < 0.05)",
         sig.index[sig.log2FoldChange > 0], sig.index[sig.log2FoldChange < 0])
add_pool("GSE99963", c, r["baseMean"])

# ---- GSE144304: rebuilt from the authors' counts, ~ condition, p < 0.01
jc = pd.read_csv(pub / "GSE144304_JoinedCounts.txt.gz", sep="\t", index_col=0)
jc = jc[~jc.index.str.startswith("__")]
s14 = samples[samples.series == "GSE144304"].copy()
s14["key"] = s14["title"].str.split("_").str[0]
col_of = {c.split("_")[0]: c for c in jc.columns}
for c in contrasts.loc[contrasts.series == "GSE144304", "contrast"]:
    dsg = design_of(c).merge(s14[["run", "key"]], left_on="sample", right_on="run")
    x = jc[[col_of[k] for k in dsg["key"]]].copy()
    x.columns = dsg["sample"].tolist()
    meta = dsg.set_index("sample")[["condition"]]
    r = deseq(x.astype(int), meta, "~ condition")
    r.index = [ens2sym.get(i.split(".")[0], "") for i in r.index]
    r = r[r.index != ""]
    sig = r[r.pvalue < 0.01]
    print(c, "rebuilt p < 0.01:", len(sig), "up", int((sig.log2FoldChange > 0).sum()), "down", int((sig.log2FoldChange < 0).sum()))
    add_list("GSE144304", c, "rebuilt (PyDESeq2 on GEO counts, ~ condition, p < 0.01)",
             sig.index[sig.log2FoldChange > 0], sig.index[sig.log2FoldChange < 0])
    add_pool("GSE144304", c, r["baseMean"])

# ---- GSE120862: rebuilt from the authors' normalised counts, ~ subject + condition, BH p < 0.05
nc = pd.read_csv(pub / "GSE120862_DESeq_normalized_counts.txt.gz", sep="\t", index_col=0)
s12 = samples[samples.series == "GSE120862"].set_index("run")
for c in contrasts.loc[contrasts.series == "GSE120862", "contrast"]:
    dsg = design_of(c)
    x = nc[[s12.loc[s, "title"] for s in dsg["sample"]]].round().astype(int)
    x.columns = dsg["sample"].tolist()
    meta = dsg.set_index("sample")[["subject", "condition"]]
    r = deseq(x, meta, "~ subject + condition")
    sig = r[r.padj < 0.05]
    print(c, "rebuilt padj < 0.05:", len(sig), "up", int((sig.log2FoldChange > 0).sum()), "down", int((sig.log2FoldChange < 0).sum()))
    add_list("GSE120862", c, "rebuilt (PyDESeq2 on GEO normalised counts, ~ subject + condition, padj < 0.05)",
             sig.index[sig.log2FoldChange > 0], sig.index[sig.log2FoldChange < 0])
    add_pool("GSE120862", c, r["baseMean"])

pd.DataFrame(rows, columns=["series", "contrast", "source", "direction", "gene"]).drop_duplicates().to_csv(
    out / "published_genes.tsv", sep="\t", index=False)
pd.DataFrame(pools, columns=["series", "contrast", "gene", "baseMean"]).to_csv(out / "gene_basemean.tsv", sep="\t", index=False)
print("wrote", out)
