#!/usr/bin/env python3
"""Gene names reached by each route of the human contrasts (notebook 361, added 2026-11-03).

Usage: human_genes.py <contrasts_dir> <contrasts.tsv> <gencode.id_map.tsv> <out_prefix>

A route's sequences (SPLASH compactors, KaMRaT contigs) were translated in six frames
(peptides of 20 or more residues) and matched by sourmash to GENCODE v50 translations; a match
counts when its parasail alignment has E <= 1e-5 and covers >= 50 % of the peptide
(`supported` in align.gencode.parquet), the Botryllus rule. A gene is reached by a route when
one of the route's supported peptides aligns to one of its proteins. The gene name is field 7
of the GENCODE header (ENSP|ENST|ENSG|OTTHUMG|OTTHUMT|transcript|gene|length).

Routes and thresholds:
  splash_compactors       SPLASH anchors, corrected p <= 0.05, extended to compactors (no direction)
  kamrat_ttest_corrected  KaMRaT contigs, KaMRaT's corrected t-test p <= 0.05
  kamrat_ttest_p05        KaMRaT contigs, Welch t-test on log2(count + 1), uncorrected p < 0.05
  kamrat_deseq2_p05       KaMRaT contig counts + PyDESeq2, uncorrected p < 0.05
  kamrat_deseq2_corrected the same, BH-corrected p <= 0.05 over all fitted contigs
Direction: up = higher in the study's second group ("PEC" in the design files: after, exercised
leg, frail, old, trained). For KaMRaT's corrected contigs it is the sign of the difference in
mean log2(count + 1), the same quantity as the t-test route.

Writes <out_prefix>.tsv (series, contrast, route, direction, gene, gene_id, peptides, sequences)
and <out_prefix>_summary.tsv (sequences kept, sequences with a supported peptide, genes reached,
per contrast, route and direction).
"""
import sys
from pathlib import Path

import numpy as np
import polars as pl

cdir, contrasts_tsv, id_map, prefix = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
contrasts = pl.read_csv(contrasts_tsv, separator="\t")
design_dir = Path(contrasts_tsv).parent
genes = (pl.read_csv(id_map, separator="\t", columns=["stable_id", "header"], quote_char=None)
         .with_columns(pl.col("header").str.split("|").alias("f"))
         .select("stable_id", pl.col("f").list.get(6).alias("gene"),
                 pl.col("f").list.get(2).str.replace(r"\.\d+$", "").alias("gene_id")))

rows, summ = [], []


def add(series, c, route, seqs, peps, align):
    """seqs: seq_id, direction; peps: peptide_id, seq_id; align: query, target (supported only)."""
    for d in sorted(seqs["direction"].unique().to_list()):
        s = seqs.filter(pl.col("direction") == d)
        p = peps.join(s.select("seq_id"), on="seq_id")
        a = (align.join(p, left_on="query", right_on="peptide_id")
             .join(genes, left_on="target", right_on="stable_id"))
        g = (a.group_by("gene", "gene_id")
             .agg(pl.col("query").n_unique().alias("peptides"), pl.col("seq_id").n_unique().alias("sequences"))
             .sort("gene"))
        summ.append(dict(series=series, contrast=c, route=route, direction=d, sequences_kept=s.height,
                         sequences_with_peptide=p["seq_id"].n_unique(),
                         sequences_with_supported_match=a["seq_id"].n_unique(), genes=g.height))
        rows.append(g.with_columns(pl.lit(series).alias("series"), pl.lit(c).alias("contrast"),
                                   pl.lit(route).alias("route"), pl.lit(d).alias("direction")))


for r in contrasts.iter_rows(named=True):
    c, series = r["contrast"], r["series"]
    dsg = pl.read_csv(design_dir / f"contrast_{c}.design.tsv", separator="\t")
    is_pec = {s: cond == "PEC" for s, cond in zip(dsg["sample"], dsg["condition"])}
    for src in ("compactors", "kamrat_ttest", "kamrat_p05", "kamrat_deseq2_p05"):
        sd = cdir / c / f"search_{src}"
        if not (sd / "search.done").exists():
            print(c, src, "no finished search")
            continue
        pt = sd / "pep.peptides.tsv"
        pep = pl.read_csv(pt, separator="\t", infer_schema_length=0) if pt.exists() else pl.DataFrame()
        al = sd / "align.gencode.parquet"
        align = (pl.read_parquet(al, columns=["query", "target", "supported"]).filter(pl.col("supported"))
                 .select("query", "target") if al.exists() else pl.DataFrame(schema={"query": pl.String, "target": pl.String}))
        # every kept sequence, with its id as translate numbered it (c1, c2, ... in file order)
        if src == "compactors":
            seqs = pl.read_csv(cdir / c / "compactors" / "compactors.tsv", separator="\t", infer_schema_length=0)
            seqs = seqs.with_row_index("i", 1).select((pl.lit("c") + pl.col("i").cast(pl.String)).alias("seq_id"),
                                                     pl.lit("none").alias("direction"))
        else:
            sub = {"kamrat_ttest": "kamrat_ttest", "kamrat_p05": "kamrat_p05",
                   "kamrat_deseq2_p05": "kamrat_deseq2_p05"}[src]
            t = pl.read_csv(cdir / c / sub / "contigs.tsv", separator="\t", infer_schema_length=10000)
            t = t.with_row_index("i", 1).with_columns((pl.lit("c") + pl.col("i").cast(pl.String)).alias("seq_id"))
            if src == "kamrat_ttest":
                samples = [s for s in dsg["sample"] if s in t.columns]
                assert len(samples) == dsg.height, "design samples missing from KaMRaT contigs"
                x = np.log2(t.select(samples).to_numpy().astype(float) + 1)
                pec = np.array([is_pec[s] for s in samples])
                diff = x[:, pec].mean(1) - x[:, ~pec].mean(1)
                t = t.with_columns(pl.Series("direction", np.where(diff > 0, "up", "down")))
        if src == "compactors":
            add(series, c, "splash_compactors", seqs, pep.select("peptide_id", "seq_id") if pep.height else
                pl.DataFrame(schema={"peptide_id": pl.String, "seq_id": pl.String}), align)
            continue
        pp = pep.select("peptide_id", "seq_id") if pep.height else pl.DataFrame(schema={"peptide_id": pl.String, "seq_id": pl.String})
        if src == "kamrat_ttest":
            add(series, c, "kamrat_ttest_corrected", t.select("seq_id", "direction"), pp, align)
        elif src == "kamrat_p05":
            add(series, c, "kamrat_ttest_p05", t.select("seq_id", "direction"), pp, align)
        else:
            add(series, c, "kamrat_deseq2_p05", t.select("seq_id", "direction"), pp, align)
            add(series, c, "kamrat_deseq2_corrected",
                t.filter(pl.col("q_bh_all") <= 0.05).select("seq_id", "direction"), pp, align)
        print(c, src, "done")

cols = ["series", "contrast", "route", "direction", "gene", "gene_id", "peptides", "sequences"]
out = pl.concat(rows).select(cols) if rows else pl.DataFrame(schema={k: pl.String for k in cols})
out.write_csv(f"{prefix}.tsv", separator="\t")
s = pl.DataFrame(summ)
s.write_csv(f"{prefix}_summary.tsv", separator="\t")
with pl.Config(tbl_rows=200, tbl_cols=20, tbl_width_chars=200):
    print(s)
