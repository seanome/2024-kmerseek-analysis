#!/usr/bin/env python3
"""Contig lengths for each KaMRaT merge setting of one contrast (notebook 361).

Usage: kamrat_merge_lengths.py <contrast_dir> <out.tsv>

<contrast_dir> holds <setting>/contigs_all.tsv and <setting>/kept/contigs.tsv (uncorrected p < 0.05).
Writes contigs, bases, mean length, and the share of 31-base contigs (one k-mer, not merged) and of
contigs of 60+ bases (the shortest that can hold a peptide of 20 residues), for all contigs and kept.
"""
import sys
from pathlib import Path

import polars as pl

contrast_dir, out_tsv = Path(sys.argv[1]), sys.argv[2]


def stats(path, tag):
    L = pl.read_csv(path, separator="\t", columns=[0], infer_schema_length=None).to_series().str.len_chars()
    return {f"{tag}_contigs": len(L), f"{tag}_bases": int(L.sum()), f"{tag}_mean": round(float(L.mean()), 1),
            f"{tag}_one_kmer_pct": round(100 * float((L == 31).mean()), 1), f"{tag}_ge60": int((L >= 60).sum()),
            f"{tag}_ge60_pct": round(100 * float((L >= 60).mean()), 1)}


rows = [dict(contrast=contrast_dir.name, setting=d.name, **stats(d / "contigs_all.tsv", "all"), **stats(d / "kept" / "contigs.tsv", "kept"))
        for d in sorted(contrast_dir.iterdir()) if (d / "kept" / "contigs.tsv").exists()]
t = pl.DataFrame(rows)
t.write_csv(out_tsv, separator="\t")
with pl.Config(tbl_rows=20, tbl_cols=20, tbl_width_chars=220):
    print(t)
