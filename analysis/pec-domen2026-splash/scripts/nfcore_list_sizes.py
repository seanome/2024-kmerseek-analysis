#!/usr/bin/env python3
"""Genes passing each cutoff in nf-core's and the paper's DESeq2, by contrast (notebook 361).

Usage: nfcore_list_sizes.py <deseq_dir> <paper_deseq_all.parquet> <out.tsv>

Expressed means baseMean >= 10, chosen after seeing that genes with few reads rarely reach p <= 0.05
whatever their change. Chance alone puts about 5 % of genes at p <= 0.05. padj05_expressed
corrects over expressed genes only, which is the kindest correction for finding genes.
"""
import sys

import pandas as pd
from statsmodels.stats.multitest import multipletests

deseq_dir, paper_pq, out = sys.argv[1:]
CONTRASTS = ["young_2h", "young_24h", "old_24h", "young_13_49d", "young_132d", "old_41_93d"]
paper = pd.read_parquet(paper_pq, columns=["contrast", "baseMean", "pvalue", "padj"])


def counts(d, source):
    t = d[d.pvalue.notna()]
    e = t[t.baseMean >= 10]
    return dict(source=source, genes_tested=len(t), p05=int((t.pvalue <= 0.05).sum()), padj05=int((t.padj < 0.05).sum()),
                pct_p05_all=round(100 * (t.pvalue <= 0.05).mean(), 2), genes_expressed=len(e),
                pct_p05_expressed=round(100 * (e.pvalue <= 0.05).mean(), 2),
                padj05_expressed=int(multipletests(e.pvalue, method="fdr_bh")[0].sum()))


rows = []
for c in CONTRASTS:
    d = pd.read_csv(f"{deseq_dir}/{c}/tables/differential/rnaseq/{c}_pec_domen2026.deseq2.results.tsv", sep="\t")
    p05 = (d.pvalue <= 0.05) & (d.log2FoldChange.abs() > 1)
    rows.append(dict(contrast=c, **counts(d, "nf-core"), p05_2fold=int(p05.sum())))
    rows.append(dict(contrast=c, **counts(paper[paper.contrast == c], "paper"), p05_2fold=None))
t = pd.DataFrame(rows)
t.to_csv(out, sep="\t", index=False)
print(t.to_string(index=False))
