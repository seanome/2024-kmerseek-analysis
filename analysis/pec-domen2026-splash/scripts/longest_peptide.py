#!/usr/bin/env python3
"""Keep each sequence's longest stop-free peptide from `pec-tools translate` output.

Usage: longest_peptide.py <prefix.peptides.tsv> <out.faa>
Ties are broken by the smaller start coordinate, then strand '+' first. The FASTA name is the
input sequence ID, so matches can be joined back to transcripts.
"""
import sys

import polars as pl

pep = pl.read_csv(sys.argv[1], separator="\t", infer_schema_length=0).with_columns(
    pl.col("aa_length").cast(pl.Int64), pl.col("nt_start").cast(pl.Int64))
best = (pep.sort(["seq_id", "aa_length", "nt_start", "strand"], descending=[False, True, False, False])
        .group_by("seq_id", maintain_order=True).head(1))
with open(sys.argv[2], "w") as f:
    for sid, aa in zip(best["seq_id"], best["peptide"]):
        f.write(f">{sid}\n{aa}\n")
print(f"sequences {best.height}")
