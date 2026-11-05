#!/usr/bin/env python3
"""How many nf-core DESeq2 genes each route reaches, going the same way (notebook 361).

Usage: nfcore_recovery.py <contrasts_dir> <deseq_dir> <proteome.id_map.tsv> <out_dir> [<gene_counts_dir>]

The reference is nf-core/rnaseq 3.27.0 (STAR + RSEM, April 2024 genome) then
nf-core/differentialabundance 2.0.0 (DESeq2), one results table per contrast in <deseq_dir>, named
<contrast>/tables/differential/rnaseq/<contrast>_pec_domen2026.deseq2.results.tsv. Two lists, up and down, for each of two cutoffs, both chosen before
the results were read:
- corrected: padj < 0.05 and 2-fold (the main reference)
- paper_cutoff: uncorrected p <= 0.05 and 2-fold (the cutoff of the paper's lists)

A route reaches a gene when one of its peptides has a supported alignment (E <= 1e-5, query
coverage >= 50 %) to that gene's protein in the new proteome, align.proteome.parquet. A new-proteome
name FUN012345_XYZ belongs to gene FUN_012345 in the GTF. Peptide direction and the test against
1,000 baseMean-matched random gene sets are those of paper_recovery.py, with the pool being genes in
the contrast's results table that have a protein. SPLASH has no direction, so it is compared with up
and down alike.

<gene_counts_dir>, when given, adds the route gene_counts: <gene_counts_dir>/<contrast>.tsv from
gene_directions.py, already named by GTF gene, so a gene is reached when it has that direction.

Writes <out_dir>/recovery.tsv, one row per contrast, route, cutoff and list.
"""
import re
import sys
from pathlib import Path

import numpy as np
import polars as pl

contrasts_dir, deseq_dir, id_map, out_dir = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4])
gene_counts_dir = Path(sys.argv[5]) if len(sys.argv) > 5 else None
out_dir.mkdir(parents=True, exist_ok=True)
N_NULL, SEED = 1000, 361
CONTRASTS = ["young_2h", "young_24h", "old_24h", "young_13_49d", "young_132d", "old_41_93d"]
SEARCH = {"splash": "search_compactors", "anchor_nominal": "search_compactors_anchor_nominal",
          "kamrat_ttest_p05": "search_kamrat_p05", "kamrat_deseq2_p05": "search_kamrat_deseq2_p05",
          "dekupl_limma_p05": "search_dekupl_limma_p05", "dekupl_deseq2_p05": "search_dekupl_deseq2_p05"}
ROUTES = list(SEARCH) + (["gene_counts"] if gene_counts_dir else [])  # new routes go last: each route's random draws depend on its place
NOT_USABLE = {("young_132d", "dekupl_limma_p05"), ("young_132d", "dekupl_deseq2_p05")}  # see paper_recovery.py
CUTOFFS = {"corrected": pl.col("padj") < 0.05, "paper_cutoff": pl.col("pvalue") <= 0.05}

proteins = (pl.read_csv(id_map, separator="\t")
            .select(pl.col("stable_id").alias("target"),
                    pl.col("original_id").map_elements(lambda s: "FUN_" + re.match(r"FUN(\d+)", s).group(1),
                                                       return_dtype=pl.Utf8).alias("gene")))
genes_with_protein = set(proteins["gene"])


def reached(contrast, channel):
    """{direction or 'any': set of genes} for one route, or None when its files are missing."""
    if (contrast, channel) in NOT_USABLE:
        return None
    if channel == "gene_counts":
        d = pl.read_csv(gene_counts_dir / f"{contrast}.tsv", separator="\t", columns=["gene", "direction"]).drop_nulls("direction")
        return {"any": set(d["gene"]), **{w: set(d.filter(pl.col("direction") == w)["gene"]) for w in ("up", "down")}}
    src = contrasts_dir / contrast / SEARCH[channel]
    aln = src / "align.proteome.parquet"
    if not aln.exists():
        return None
    hits = (pl.read_parquet(aln, columns=["query", "target", "supported"]).filter(pl.col("supported"))
            .join(proteins, on="target").select(pl.col("query").alias("peptide_id"), "gene"))
    out = {"any": set(hits["gene"])}
    if channel == "splash":
        return out
    if channel == "anchor_nominal":
        pep = pl.read_csv(src / "pep.peptides.tsv", separator="\t", columns=["peptide_id", "anchor"])
        anchors = pl.read_csv(contrasts_dir / contrast / "compactors_anchor_nominal/anchors_nominal.tsv",
                              separator="\t", columns=["anchor", "direction"])
        hd = hits.join(pep, on="peptide_id").join(anchors, on="anchor")
    else:
        pep = pl.read_csv(src / "pep.peptides.tsv", separator="\t", columns=["peptide_id", "direction"])
        hd = hits.join(pep, on="peptide_id")
    assert hd.height == hits.height, f"{contrast} {channel}: peptides without a direction"
    for d in ("up", "down"):
        out[d] = set(hd.filter(pl.col("direction") == d)["gene"])
    return out


rows = []
for c in CONTRASTS:
    res = pl.read_csv(deseq_dir / c / "tables/differential/rnaseq" / f"{c}_pec_domen2026.deseq2.results.tsv", separator="\t", null_values=["NA"], infer_schema_length=None)
    pool = (res.filter(pl.col("baseMean").is_not_null() & (pl.col("baseMean") > 0)
                       & pl.col("gene_id").is_in(list(genes_with_protein)))
            .with_columns(pl.col("baseMean").qcut(10, labels=[str(i) for i in range(10)], allow_duplicates=True)
                          .cast(pl.Utf8).alias("decile")))
    decile_of = dict(zip(pool["gene_id"], pool["decile"]))
    by_decile = {d: np.array(sorted(g)) for d, g in pool.group_by("decile").agg("gene_id").iter_rows()}
    for ci, channel in enumerate(ROUTES):
        got = reached(c, channel)
        if got is None:
            print(f"{c} {channel}: no alignment file or not usable, skipped", flush=True)
            continue
        for cutoff, keep in CUTOFFS.items():
            r_ = np.random.default_rng([SEED, ci, CONTRASTS.index(c), list(CUTOFFS).index(cutoff)])
            for direction, fold in (("up", pl.col("log2FoldChange") > 1), ("down", pl.col("log2FoldChange") < -1)):
                usable = sorted(pool.filter(keep.fill_null(False) & fold)["gene_id"])
                hit_set = got.get(direction, got["any"])
                row = dict(contrast=c, channel=channel, cutoff=cutoff, list=direction,
                           compared_with=direction if direction in got else "any", list_genes=len(usable),
                           reached=sum(g in hit_set for g in usable), reached_either_way=sum(g in got["any"] for g in usable),
                           pool_genes=pool.height, pool_fraction_reached=sum(g in hit_set for g in pool["gene_id"]) / pool.height)
                if usable:
                    need = pl.Series("d", [decile_of[g] for g in usable]).value_counts().sort("d")
                    null = np.array([sum(g in hit_set for g in np.concatenate(
                        [r_.choice(by_decile[d], size=n, replace=False) for d, n in need.iter_rows()])) / len(usable)
                        for _ in range(N_NULL)])
                    obs = row["reached"] / len(usable)
                    row |= dict(fraction_reached=obs, null_median=float(np.median(null)),
                                null_97_5=float(np.quantile(null, 0.975)), share_null_at_or_above=float((null >= obs).mean()))
                else:
                    row |= dict(fraction_reached=None, null_median=None, null_97_5=None, share_null_at_or_above=None)
                rows.append(row)
                print(row, flush=True)

t = pl.DataFrame(rows, infer_schema_length=None)
t.write_csv(out_dir / "recovery.tsv", separator="\t")
s = (t.group_by("cutoff", "channel")
     .agg(pl.col("list_genes").sum(), pl.col("reached").sum(), pl.col("reached_either_way").sum(),
          (pl.col("share_null_at_or_above") <= 0.05).sum().alias("lists_beating_random"),
          (pl.col("list_genes") > 0).sum().alias("lists_with_genes"))
     .with_columns((pl.col("reached") / pl.col("list_genes")).round(3).alias("share")).sort("cutoff", "share", descending=[False, True]))
s.write_csv(out_dir / "summary.tsv", separator="\t")
with pl.Config(tbl_rows=30, tbl_cols=12, tbl_width_chars=160):
    print(s)
