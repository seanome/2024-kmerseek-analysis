#!/usr/bin/env python3
"""Known-answer input for fig4_score_from_reads.py: the authors' gene tables in its input layout.

Usage: fig4_gene_table_check.py <geo_dir> <ensg_symbol.tsv> <samples.tsv> <out_root> [<read_counts_root>]

geo_dir holds GSE99963_GeneCoverage.tsv.gz and GSE144304_JoinedCounts.txt.gz from GEO.
ensg_symbol.tsv has no header: Ensembl gene ID without version, GENCODE v50 gene symbol in upper
case (columns 3 and 7 of the GENCODE protein FASTA headers). Counts of gene IDs that share a symbol
are added up; gene IDs with no protein-coding symbol are dropped, as the read counter cannot see
them either. Writes <out_root>/<series>/share_1/, with every read counted as assigned and a
placeholder cost file, so only the score columns of the result mean anything.

Run fig4_score_from_reads.py on out_root: its scores should rank samples as PR 212's do.

With read_counts_root (the folder of read-based counts), it also writes the gene table thinned to
the depth of each thinned read count found there: for share_<N> of a series, each sample keeps each
of its table reads with probability (reads placed in a gene from the 1/N share) / (table reads),
drawn with seed 1. This asks whether fewer reads alone change the score, with the counting method
held fixed. Added 2026-11-09, after the first read-based counts.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

geo, symbols, samples, out_root = (Path(p) for p in sys.argv[1:5])
read_counts_root = Path(sys.argv[5]) if len(sys.argv) > 5 else None
rng = np.random.default_rng(1)
COST_PLACEHOLDER = (
    "User time (seconds): 0.0\nSystem time (seconds): 0.0\n"
    "Elapsed (wall clock) time (h:mm:ss or m:ss): 0:00.00\nMaximum resident set size (kbytes): 0\n"
)


def save(g, out):
    g = g[g.sum(axis=1) > 0]
    out.mkdir(parents=True, exist_ok=True)
    g.rename_axis("gene").to_csv(out / "gene.counts.tsv", sep="\t")
    total = g.sum().to_numpy()
    pd.DataFrame({"sample": g.columns, "pairs": total, "no_hit": 0, "tied": 0, "assigned": total}).to_csv(
        out / "gene.summary.tsv", sep="\t", index=False
    )
    (out / "gene_counts.time").write_text(COST_PLACEHOLDER)


sym = pd.read_csv(symbols, sep="\t", header=None, index_col=0)[1]
sym = sym[~sym.index.duplicated()]
design = pd.read_csv(samples, sep="\t")


def write(series, table, key):
    table = table[table.index.str.startswith("ENSG")]  # GSE144304 ends in HTSeq's __no_feature style rows
    table.index = table.index.str.replace(r"\.\d+$", "", regex=True)
    d = design[design.series == series]
    table = table.rename(columns=dict(zip(key(d.title), d.run)))[list(d.run)]
    g = table.groupby(sym.reindex(table.index).to_numpy(), dropna=True).sum()
    save(g, out_root / series / "share_1")
    if read_counts_root is not None:
        for folder in sorted((read_counts_root / series).glob("share_*")):
            if folder.name == "share_1":
                continue
            placed = pd.read_csv(folder / "gene.summary.tsv", sep="\t").set_index("sample").assigned
            thin = g.copy()
            for run in thin.columns:
                keep = min(1.0, placed[run] / g[run].sum())
                thin[run] = rng.binomial(g[run].to_numpy().astype(int), keep)
            save(thin, out_root / series / folder.name)
    print(series, g.shape, "gene IDs without a protein-coding symbol:", int(sym.reindex(table.index).isna().sum()))


write("GSE99963", pd.read_csv(geo / "GSE99963_GeneCoverage.tsv.gz", sep="\t").set_index("GeneID"), lambda t: t)
t = pd.read_csv(geo / "GSE144304_JoinedCounts.txt.gz", sep="\t", index_col=0)
t.columns = t.columns.str.split("_").str[0]
write("GSE144304", t, lambda title: title.str.split("_").str[0])
