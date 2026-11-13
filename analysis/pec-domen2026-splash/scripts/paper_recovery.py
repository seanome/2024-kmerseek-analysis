#!/usr/bin/env python3
"""How many of the paper's listed genes each channel reaches, against a baseMean-matched null.

Usage: paper_recovery.py <contrasts_dir> <paper_deseq_all.parquet> <paper_named_lists.tsv>
                         <botznik.id_map.tsv> <out.tsv>

The null was fixed in design/ANALYSIS_PLAN.md (2026-10-30) before the comparison was run.

A channel reaches a gene when one of its peptides has a supported alignment (E <= 1e-5 and query
coverage >= 50 %, the `supported` column of align.botznik.parquet) to that gene's old gene-model
protein (botznik IDs g<N>.t<M>, gene g<N>). Channels:
- splash: compactors of every SPLASH anchor with corrected p <= 0.05. No direction, so it is
  compared with the up and the down list alike.
- anchor_nominal: compactors of the anchors whose anchor-count DESeq2 fit passes the paper's
  cutoff (uncorrected p <= 0.05, |log2FC| > 1; select_nominal_anchors.py). A peptide carries the
  direction of its anchor, and the up list is compared with genes reached by up anchors.
Added 2026-11-02 (Olga: "what about KaMRaT and DE-kupl here?", then "use nominal p < 0.05 for
kamrat and de-kupl" and "do KAMRAT + DeSeq2 not just a ttest"). Each has a direction per contig,
carried into pep.peptides.tsv by the search step's translate (column `direction`, up = higher in PEC):
- kamrat_nominal: KaMRaT contigs, Welch t-test on log2(count + 1), p <= 0.05 and 2-fold
  (select_kamrat_nominal.py; scored for comparison, not drawn).
- kamrat_ttest_p05: the same t-test, p < 0.05, no fold-change cutoff.
- kamrat_deseq2_p05: KaMRaT contigs tested with PyDESeq2 (kamrat_contig_counts.py,
  anchor_deseq.py, select_kamrat_deseq.py), p < 0.05, no fold-change cutoff.
- dekupl_limma_p05: DE-kupl k-mers with limma p < 0.05 (raw_pvals.txt.gz), no fold-change
  cutoff, merged into contigs per direction (select_dekupl_nominal.py).
- dekupl_deseq2_p05: the same with DE-kupl's DESeq2 test (19_dekupl_deseq2_p05.sbatch).
- kamrat_ttest_p05_reads, kamrat_deseq2_p05_reads: the reads holding those KaMRaT contigs' k-mers in
  place of the contigs (kamrat_reads.sbatch); a read carries its contig's direction. Listed last so
  that the channels above keep their random streams.
A channel whose search folder is missing is skipped. Each channel and contrast has its own random
stream (seed [361, channel number, contrast number]), and the deciles and their genes are drawn
in sorted order, so a rerun gives the same numbers. Before 2026-11-02 one stream was shared and
the decile order came from polars value_counts, which is not fixed; rerunning that version moved
12 of the 24 shares of random sets by 0.001 to 0.022 (no list crossed 0.05 or 0.004).

For each contrast, channel and list (up, down), the observed fraction of listed genes reached is
compared with 1,000 random gene sets of the same size, drawn from the genes in the paper's DESeq2
sheet of that contrast, matching the list's count in each baseMean decile. Only genes with an old
gene-model protein can be reached, so both the list and the pool of genes are limited to those.
Reported: observed fraction, null median and 2.5-97.5 % range, and the share of random sets at or
above the observed fraction.
"""
import sys
from pathlib import Path

import numpy as np
import polars as pl

contrasts_dir, paper_deseq, named_lists, id_map, out_tsv = sys.argv[1:]
contrasts_dir = Path(contrasts_dir)
N_NULL, SEED = 1000, 361
CONTRASTS = ["young_2h", "young_24h", "old_24h", "young_13_49d", "young_132d", "old_41_93d"]

bz = (pl.read_csv(id_map, separator="\t").filter(pl.col("original_id").str.contains(r"^g\d+\.t\d+$"))
      .select(pl.col("stable_id").alias("target"), pl.col("original_id").str.replace(r"\.t\d+$", "").alias("gene")))
genes_with_protein = set(bz["gene"])
deseq = pl.read_parquet(paper_deseq, columns=["Gene_ID", "baseMean", "contrast"])
lists = pl.read_csv(named_lists, separator="\t")


SEARCH = {"splash": "search_compactors", "anchor_nominal": "search_compactors_anchor_nominal",
          "kamrat_nominal": "search_kamrat_nominal", "kamrat_ttest_p05": "search_kamrat_p05",
          "kamrat_deseq2_p05": "search_kamrat_deseq2_p05", "dekupl_limma_p05": "search_dekupl_limma_p05",
          "dekupl_deseq2_p05": "search_dekupl_deseq2_p05", "kamrat_ttest_p05_reads": "search_kamrat_p05_reads",
          "kamrat_deseq2_p05_reads": "search_kamrat_deseq2_p05_reads"}


# DE-kupl's own normalization factors for young_132d are broken (2 samples 0, 4 samples 1e219 to
# 1e262; DEkupl_result/metadata/sample_conditions_full.tsv), so its limma p-values there are not
# usable and its DESeq2 test stops ("all(value > 0) is not TRUE"). Not scored (2026-11-02).
NOT_USABLE = {("young_132d", "dekupl_limma_p05"), ("young_132d", "dekupl_deseq2_p05")}


def reached(contrast, channel):
    """{direction or 'any': set of genes} for one channel, or None when its files are missing."""
    if (contrast, channel) in NOT_USABLE:
        return None
    src = SEARCH[channel]
    aln = contrasts_dir / contrast / src / "align.botznik.parquet"
    if not aln.exists():
        return None
    hits = (pl.read_parquet(aln, columns=["query", "target", "supported"]).filter(pl.col("supported"))
            .join(bz, on="target").select(pl.col("query").alias("peptide_id"), "gene"))
    out = {"any": set(hits["gene"])}
    if channel != "splash":
        if channel == "anchor_nominal":
            pep = pl.read_csv(contrasts_dir / contrast / src / "pep.peptides.tsv", separator="\t",
                              columns=["peptide_id", "anchor"])
            anchors = pl.read_csv(contrasts_dir / contrast / "compactors_anchor_nominal/anchors_nominal.tsv",
                                  separator="\t", columns=["anchor", "direction"])
            hd = hits.join(pep, on="peptide_id").join(anchors, on="anchor")
        else:
            pep = pl.read_csv(contrasts_dir / contrast / src / "pep.peptides.tsv", separator="\t",
                              columns=["peptide_id", "direction"])
            hd = hits.join(pep, on="peptide_id")
        assert set(hd["direction"].unique()) <= {"up", "down"}, (contrast, channel, hd["direction"].unique())
        assert hd.height == hits.height, f"{contrast} {channel}: peptides without a direction"
        for d in ("up", "down"):
            out[d] = set(hd.filter(pl.col("direction") == d)["gene"])
    return out


rows = []
for c in CONTRASTS:
    pool = (deseq.filter((pl.col("contrast") == c) & pl.col("baseMean").is_not_null()
                         & pl.col("Gene_ID").is_in(list(genes_with_protein)))
            .with_columns(pl.col("baseMean").qcut(10, labels=[str(i) for i in range(10)], allow_duplicates=True)
                          .cast(pl.Utf8).alias("decile")))
    decile_of = dict(zip(pool["Gene_ID"], pool["decile"]))
    by_decile = {d: np.array(sorted(g)) for d, g in pool.group_by("decile").agg("Gene_ID").iter_rows()}
    for ci, channel in enumerate(SEARCH):
        r_ = np.random.default_rng([SEED, ci, CONTRASTS.index(c)])
        got = reached(c, channel)
        if got is None:
            print(f"{c} {channel}: no alignment file or not usable, skipped", flush=True)
            continue
        for direction in ("up", "down"):
            genes = set(lists.filter((pl.col("contrast") == c) & (pl.col("direction") == direction))["gene_id"])
            usable = sorted(g for g in genes if g in decile_of)
            hit_set = got.get(direction, got["any"])
            obs = sum(g in hit_set for g in usable) / len(usable) if usable else float("nan")
            need = pl.Series([decile_of[g] for g in usable]).value_counts() if usable else None
            if usable:
                need = need.sort(need.columns[0])
            null = []
            for _ in range(N_NULL):
                draw = np.concatenate([r_.choice(by_decile[d], size=n, replace=False)
                                       for d, n in need.iter_rows()]) if usable else []
                null.append(sum(g in hit_set for g in draw) / len(usable) if usable else float("nan"))
            null = np.array(null)
            rows.append(dict(
                contrast=c, channel=channel, list=direction, compared_with=direction if direction in got else "any",
                list_genes=len(genes), list_genes_with_protein=len(usable),
                reached=sum(g in hit_set for g in usable), fraction_reached=obs,
                reached_either_way=sum(g in got["any"] for g in usable),
                null_median=float(np.median(null)), null_2_5=float(np.quantile(null, 0.025)),
                null_97_5=float(np.quantile(null, 0.975)), share_null_at_or_above=float((null >= obs).mean()),
                pool_genes=pool.height, pool_fraction_reached=sum(g in hit_set for g in pool["Gene_ID"]) / pool.height,
            ))
            print(rows[-1], flush=True)

pl.DataFrame(rows).write_csv(out_tsv, separator="\t")
print("wrote", out_tsv)
