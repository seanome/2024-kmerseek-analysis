#!/usr/bin/env python3
"""One line of totals from a thinned gene-counts run's recovery table (notebook 361).

Usage: thin_reached.py <recovery.tsv> <out.reached.tsv>

Reads either table: cheap_route_score.py on the paper's lists (one channel, which holds the gene-counts
directions whatever its name says), or nfcore_recovery.py on nf-core's lists (several channels and
cutoffs: keeps gene_counts at the paper's cutoff). Lists with no genes are left out. A list beats random
when 5 % or fewer of the random gene sets reach as many.
Writes genes, reached, lists_beat_random and lists, read by plot_gain_vs_cost.py.
"""
import sys

import pandas as pd

recovery_tsv, out_tsv = sys.argv[1:]
r = pd.read_csv(recovery_tsv, sep="\t")
genes = "list_genes_with_protein"
if "cutoff" in r.columns:
    r = r[(r.cutoff == "paper_cutoff") & (r.channel == "gene_counts")]
    genes = "list_genes"
assert r.channel.nunique() == 1, recovery_tsv
r = r[r[genes] > 0]
pd.DataFrame([dict(genes=r[genes].sum(), reached=r.reached.sum(), lists_beat_random=(r.share_null_at_or_above <= 0.05).sum(),
                   lists=len(r))]).to_csv(out_tsv, sep="\t", index=False)
