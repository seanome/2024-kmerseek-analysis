#!/usr/bin/env python3
"""KaMRaT contigs whose PyDESeq2 fit has uncorrected p < 0.05, with no fold-change cutoff.

Usage: select_kamrat_deseq.py <deseq_<design>.parquet> <out_dir>

The parquet is anchor_deseq.py run on kamrat_contig_counts.py's table (column `anchor` holds the
contig sequence). Added 2026-11-02 at Olga's request: "do KAMRAT + DeSeq2 not just a ttest" and
"use nominal p < 0.05 for kamrat and de-kupl". Direction: up = log2 fold change > 0 = higher in
PEC (anchor_deseq.py tests PEC against control). Contigs whose p is missing (all-zero or
Cook's-distance outliers) are not kept.

Writes <out_dir>/contigs.tsv (contig, direction, log2FoldChange, pvalue, q_bh_all, baseMean; the search
step reads the column `contig`) and <out_dir>/summary.json.
"""
import json
import sys
from pathlib import Path

import polars as pl

deseq, out = sys.argv[1], Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
d = pl.read_parquet(deseq, columns=["anchor", "baseMean", "log2FoldChange", "pvalue", "q_bh_all"])
keep = (
    d.filter(pl.col("pvalue") < 0.05)
    .with_columns(pl.when(pl.col("log2FoldChange") > 0).then(pl.lit("up")).otherwise(pl.lit("down")).alias("direction"))
    .select(pl.col("anchor").alias("contig"), "direction", "log2FoldChange", "pvalue", "q_bh_all", "baseMean")
    .sort("contig")
)
keep.write_csv(out / "contigs.tsv", separator="\t")
s = {"contigs_fitted": d.height, "pvalue_missing": d["pvalue"].is_null().sum() + d["pvalue"].is_nan().sum(),
     "p_lt_0.05": keep.height, "up": keep.filter(pl.col("direction") == "up").height,
     "down": keep.filter(pl.col("direction") == "down").height,
     "q_bh_le_0.05": d.filter(pl.col("q_bh_all") <= 0.05).height,
     "source": str(deseq)}
(out / "summary.json").write_text(json.dumps(s, indent=1) + "\n")
print(json.dumps(s))
