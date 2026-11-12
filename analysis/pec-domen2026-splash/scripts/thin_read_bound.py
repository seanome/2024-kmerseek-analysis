#!/usr/bin/env python3
"""How many of the paper's listed genes could still be seen with a fraction of the reads (notebook 361).

Usage: thin_read_bound.py <paper_deseq_all.parquet> <paper_named_lists.tsv> <out.tsv>

Uses only the paper's own numbers. baseMean is DESeq2's mean normalised count per sample, so a
fraction f of the reads leaves about f x baseMean reads of a gene per sample. For each listed gene
and contrast with n samples:
  seen    expected chance of 1 or more reads across the contrast, 1 - exp(-n x f x baseMean)
  enough  f x baseMean x n / 2 >= 10, about 10 reads per group, a rough least for calling a direction
This is a best case: every read is taken to land on its gene, and a sample's real depth is ignored.
Depths within a contrast differ by up to 1,000-fold, so real counts are less even than this.
"""
import sys

import numpy as np
import polars as pl

deseq, lists, out = sys.argv[1:]
N = {"young_2h": 12, "young_24h": 12, "old_24h": 4, "young_13_49d": 8, "young_132d": 6, "old_41_93d": 5}
d = pl.read_parquet(deseq, columns=["Gene_ID", "contrast", "baseMean"])
j = pl.read_csv(lists, separator="\t").join(d, left_on=["gene_id", "contrast"], right_on=["Gene_ID", "contrast"], how="left")
assert j["baseMean"].null_count() == 0
m, n = j["baseMean"].to_numpy(), np.array([N[c] for c in j["contrast"]])
rows = [dict(fraction_of_reads=f, list_genes=len(m), expected_seen=round(float((1 - np.exp(-n * m * f)).sum())),
             about_10_reads_per_group=int((m * f * n / 2 >= 10).sum())) for f in (1, 0.1, 0.01, 0.001)]
t = pl.DataFrame(rows)
t.write_csv(out, separator="\t")
print(t)
