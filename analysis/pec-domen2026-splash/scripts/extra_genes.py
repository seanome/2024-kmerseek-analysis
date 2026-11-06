#!/usr/bin/env python3
"""What the reference-free routes find beyond the paper, and which genes they could help name.

Usage: extra_genes.py <contrasts_dir> <db_dir> <paper_deseq_all.parquet> <paper_named_lists.tsv> <out_dir>

For each contrast and route (splash: compactors of SPLASH anchors at corrected p <= 0.05;
anchor_nominal: compactors of anchors at the paper's cutoff in the anchor-count DESeq2 fit), each
translated peptide (20 residues or more) was aligned to three protein sets. A match counts when
the alignment check passes (`supported`: E <= 1e-5 and query coverage >= 50 %):
- botznik: the old gene-model proteins the paper used (gene g<N>);
- proteome: the new Botryllus proteome (FUN<N>, some with a gene name after the underscore);
- swissprot: UniProt Swiss-Prot 2026_03.

Outputs, in <out_dir>:
- extra_genes_summary.tsv, per contrast and route: peptides; genes reached; how many are on the
  paper's lists for that contrast; how many the paper left unnamed (Gene_Name equal to Gene_ID);
  how many of those get a Swiss-Prot name from us; how many named genes get the same name from
  us (a check of the naming); peptides that match no old gene model but match the new proteome,
  and peptides that match neither but match Swiss-Prot.
- name_candidates.tsv: one row per contrast, route and old gene the paper left unnamed, with the
  Swiss-Prot gene name most of its peptides point to, how many peptides agree, and the best
  alignment (identity, E-value, Swiss-Prot entry and species).
- name_check.tsv: the same for genes the paper did name, with whether our name matches.
- no_gene_model.tsv: peptides with a Swiss-Prot or new-proteome match but no old gene-model match.

A gene's Swiss-Prot name comes from the peptides that match both that gene and Swiss-Prot. Each
such peptide votes for the gene name (GN=) of its best Swiss-Prot match by bit score. A second,
independent name comes the same way from the new proteome, whose protein IDs carry a gene name
after the underscore when it has one (FUN000002_NLRC5). Names are compared after upper-casing and
dropping non-letters and non-digits; "same family" means the first three characters agree
(TUBA1C and TUBA1A), a loose check, since unrelated genes can share them.
"""
import re
import sys
from pathlib import Path

import polars as pl

contrasts_dir, db_dir, paper_deseq, named_lists, out_dir = map(Path, sys.argv[1:])
out_dir.mkdir(parents=True, exist_ok=True)
CONTRASTS = ["young_2h", "young_24h", "old_24h", "young_13_49d", "young_132d", "old_41_93d"]
ROUTES = {"splash": "search_compactors", "anchor_nominal": "search_compactors_anchor_nominal"}
COLS = ["query", "target", "bit_score", "evalue", "pct_identity", "query_coverage", "supported"]

bz = (pl.read_csv(db_dir / "botznik.id_map.tsv", separator="\t")
      .filter(pl.col("original_id").str.contains(r"^g\d+\.t\d+$"))
      .select(pl.col("stable_id").alias("target"), pl.col("original_id").str.replace(r"\.t\d+$", "").alias("gene")))
prot = (pl.read_csv(db_dir / "proteome.id_map.tsv", separator="\t")
        .select(pl.col("stable_id").alias("target"), pl.col("original_id").alias("proteome_id"),
                pl.col("original_id").str.extract(r"^FUN\d+_(?:FUN\d+)?(.*)$").replace("", None).alias("prot_gene")))
sp = (pl.read_csv(db_dir / "swissprot.id_map.tsv", separator="\t", columns=["stable_id", "original_id", "header"])
      .select(pl.col("stable_id").alias("target"),
              pl.col("original_id").str.split("|").list.get(1).alias("sp_accession"),
              pl.col("header").str.extract(r"GN=(\S+)").alias("sp_gene"),
              pl.col("header").str.extract(r"^\S+ (.*?) OS=").alias("sp_protein"),
              pl.col("header").str.extract(r"OS=(.*?) OX=").alias("sp_species")))
paper = (pl.read_parquet(paper_deseq, columns=["Gene_ID", "Gene_Name", "Gene_Note", "contrast"])
         .unique(["Gene_ID"]).select("Gene_ID", "Gene_Name", "Gene_Note")
         .with_columns((pl.col("Gene_Name") == pl.col("Gene_ID")).alias("paper_unnamed")))
lists = pl.read_csv(named_lists, separator="\t")


def norm(name):
    """Gene symbol in a form two species' symbols can be compared in (Cldn1 ~ CLDN1)."""
    return None if name is None else re.sub(r"[^A-Z0-9]", "", name.upper())


def family(key):
    return None if key is None else key[:3]


def best_per_query(aln):
    return aln.filter(pl.col("supported")).sort("bit_score", descending=True).unique("query", keep="first")


summary, cands, checks, nogm = [], [], [], []
for c in CONTRASTS:
    listed = set(lists.filter(pl.col("contrast") == c)["gene_id"])
    for route, src in ROUTES.items():
        d = contrasts_dir / c / src
        if not (d / "align.botznik.parquet").exists():
            print(f"{c} {route}: no alignments, skipped", flush=True)
            continue
        n_pep = pl.scan_csv(d / "pep.peptides.tsv", separator="\t").select(pl.len()).collect().item()
        b = (pl.read_parquet(d / "align.botznik.parquet", columns=COLS).filter(pl.col("supported"))
             .join(bz, on="target").select("query", "gene").unique())
        p = best_per_query(pl.read_parquet(d / "align.proteome.parquet", columns=COLS)).join(prot, on="target")
        s = best_per_query(pl.read_parquet(d / "align.swissprot.parquet", columns=COLS)).join(sp, on="target")

        genes = b.select("gene").unique().join(paper, left_on="gene", right_on="Gene_ID", how="left")
        # each peptide that hits the gene and Swiss-Prot votes for its best Swiss-Prot gene name
        votes = (b.join(s.select("query", "sp_gene", "sp_accession", "sp_protein", "sp_species",
                                 "pct_identity", "evalue", "bit_score"), on="query")
                 .filter(pl.col("sp_gene").is_not_null())
                 .with_columns(pl.col("sp_gene").map_elements(norm, return_dtype=pl.Utf8).alias("sp_key")))
        per_gene = (votes.group_by("gene", "sp_key").agg(
                        pl.len().alias("peptides_agreeing"),
                        pl.col("sp_gene").first(),
                        pl.col("sp_protein").sort_by("bit_score", descending=True).first(),
                        pl.col("sp_species").sort_by("bit_score", descending=True).first(),
                        pl.col("sp_accession").sort_by("bit_score", descending=True).first(),
                        pl.col("pct_identity").max().alias("best_pct_identity"),
                        pl.col("evalue").min().alias("best_evalue"))
                    .join(votes.group_by("gene").agg(pl.len().alias("peptides_with_name")), on="gene")
                    .sort(["gene", "peptides_agreeing", "best_pct_identity"], descending=[False, True, True])
                    .unique("gene", keep="first")
                    .join(genes, on="gene", how="left")
                    .with_columns(pl.lit(c).alias("contrast"), pl.lit(route).alias("route"),
                                  pl.col("gene").is_in(list(listed)).alias("on_paper_list")))
        # the new proteome's name for the gene, by the same vote
        pvotes = (b.join(p.filter(pl.col("prot_gene").is_not_null()).select("query", "prot_gene"), on="query")
                  .with_columns(pl.col("prot_gene").map_elements(norm, return_dtype=pl.Utf8).alias("prot_key")))
        prot_name = (pvotes.group_by("gene", "prot_key").agg(pl.len().alias("prot_peptides_agreeing"),
                                                            pl.col("prot_gene").first())
                     .sort(["gene", "prot_peptides_agreeing"], descending=[False, True]).unique("gene", keep="first"))
        per_gene = (per_gene.join(prot_name, on="gene", how="full", coalesce=True)
                    .with_columns(pl.col("paper_unnamed").fill_null(pl.col("gene").is_in(
                                      list(paper.filter(pl.col("paper_unnamed"))["Gene_ID"]))),
                                  pl.lit(c).alias("contrast"), pl.lit(route).alias("route"),
                                  pl.col("gene").is_in(list(listed)).alias("on_paper_list"))
                    .drop("Gene_Name", "Gene_Note").join(paper.drop("paper_unnamed"), left_on="gene",
                                                         right_on="Gene_ID", how="left")
                    .with_columns((pl.col("sp_key") == pl.col("prot_key")).alias("swissprot_and_proteome_agree")))
        unnamed = per_gene.filter(pl.col("paper_unnamed") & (pl.col("sp_key").is_not_null() | pl.col("prot_key").is_not_null()))
        paper_key = pl.col("Gene_Name").map_elements(norm, return_dtype=pl.Utf8)
        named = (per_gene.filter(~pl.col("paper_unnamed"))
                 .with_columns(paper_key.alias("paper_key"))
                 .with_columns((pl.col("paper_key") == pl.col("sp_key")).alias("same_name"),
                               (pl.col("paper_key").str.slice(0, 3) == pl.col("sp_key").str.slice(0, 3)).alias("same_family"),
                               (pl.col("paper_key") == pl.col("prot_key")).alias("proteome_same_name"),
                               (pl.col("paper_key").str.slice(0, 3) == pl.col("prot_key").str.slice(0, 3)).alias("proteome_same_family")))
        cands.append(unnamed)
        checks.append(named)

        in_gm = set(b["query"])
        no_gm = (pl.concat([
                    p.filter(~pl.col("query").is_in(list(in_gm))).select("query", pl.lit("proteome").alias("db"),
                                                                         pl.col("proteome_id").alias("match"),
                                                                         "pct_identity", "evalue"),
                    s.filter(~pl.col("query").is_in(list(in_gm))).select("query", pl.lit("swissprot").alias("db"),
                                                                         (pl.col("sp_gene").fill_null("") + " " +
                                                                          pl.col("sp_protein").fill_null("") + " (" +
                                                                          pl.col("sp_species").fill_null("") + ")").alias("match"),
                                                                         "pct_identity", "evalue")])
                 .with_columns(pl.lit(c).alias("contrast"), pl.lit(route).alias("route")))
        nogm.append(no_gm)
        no_gm_pep = set(no_gm["query"])
        prot_only = set(no_gm.filter(pl.col("db") == "proteome")["query"])
        sp_only = no_gm_pep - prot_only

        row = dict(
            contrast=c, route=route, peptides=n_pep, peptides_matching_old_gene=len(in_gm),
            genes_reached=genes.height,
            genes_on_paper_list=int(genes["gene"].is_in(list(listed)).sum()),
            genes_not_on_paper_list=int((~genes["gene"].is_in(list(listed))).sum()),
            genes_paper_unnamed=int(genes["paper_unnamed"].fill_null(False).sum()),
            unnamed_genes_given_swissprot_name=int(unnamed["sp_key"].is_not_null().sum()),
            unnamed_genes_given_proteome_name=int(unnamed["prot_key"].is_not_null().sum()),
            unnamed_genes_both_names_agree=int(unnamed["swissprot_and_proteome_agree"].fill_null(False).sum()),
            named_genes_with_swissprot_name=int(named["sp_key"].is_not_null().sum()),
            named_same_name_swissprot=int(named["same_name"].fill_null(False).sum()),
            named_same_family_swissprot=int(named["same_family"].fill_null(False).sum()),
            named_genes_with_proteome_name=int(named["prot_key"].is_not_null().sum()),
            named_same_name_proteome=int(named["proteome_same_name"].fill_null(False).sum()),
            named_same_family_proteome=int(named["proteome_same_family"].fill_null(False).sum()),
            peptides_no_old_gene_but_new_proteome=len(prot_only),
            peptides_no_botryllus_gene_but_swissprot=len(sp_only),
        )
        summary.append(row)
        print(row, flush=True)

pl.DataFrame(summary).write_csv(out_dir / "extra_genes_summary.tsv", separator="\t")
order = ["contrast", "route", "gene", "on_paper_list", "sp_gene", "sp_protein", "sp_species", "sp_accession",
         "peptides_agreeing", "peptides_with_name", "best_pct_identity", "best_evalue",
         "prot_gene", "prot_peptides_agreeing", "swissprot_and_proteome_agree"]
pl.concat(cands).select(order).sort(["contrast", "route", "peptides_agreeing"], descending=[False, False, True]) \
    .write_csv(out_dir / "name_candidates.tsv", separator="\t")
pl.concat(checks).select(order[:3] + ["Gene_Name", "same_name", "same_family", "proteome_same_name",
                                      "proteome_same_family"] + order[4:]) \
    .write_csv(out_dir / "name_check.tsv", separator="\t")
pl.concat(nogm).write_parquet(out_dir / "no_gene_model.parquet")
print("wrote", out_dir)
