#!/usr/bin/env python3
"""Proposed gene names and a figure from extra_genes.py's outputs.

Usage: plot_extra_genes.py <extra_genes_dir> <paper_named_lists.tsv> <out_stem>

Writes <extra_genes_dir>/gene_names_proposed.tsv: one row per old gene model reached by a
changed sequence in any contrast whose name in the paper's sheet is missing (Gene_Name equal to
Gene_ID), a bare number, or a date (Excel turns symbols such as SEPT2 into dates). Columns: the
paper's name and which of those three it is, the name from the new Botryllus proteome and how
many peptides agree on it, the Swiss-Prot name with its protein, species, identity and E-value,
the contrasts and routes that reached the gene, and the paper lists (contrast:direction) the
gene is on, if any.

Figure, two panels:
a. Genes reached by SPLASH in each contrast, split into those on the paper's lists for that
   contrast and those not.
b. How often our name matches the paper's where the paper has a real name (a check of the method),
   for each naming source: same symbol, and same first three characters (same family, loosely).
Greys only, so no colour here repeats a colour that means a route in the other figures.
Prints every number shown.
"""
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "notebooks"))
import pubfig as pf  # noqa: E402

d, lists_tsv, out_stem = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
summary = pl.read_csv(d / "extra_genes_summary.tsv", separator="\t")
cands = pl.read_csv(d / "name_candidates.tsv", separator="\t", infer_schema_length=0)
check = pl.read_csv(d / "name_check.tsv", separator="\t", infer_schema_length=0)


def kind(col):
    return (pl.when(pl.col(col).str.contains(r"^\d{4}-\d{2}-\d{2}$")).then(pl.lit("Excel date"))
            .when(pl.col(col).str.contains(r"^\d+$")).then(pl.lit("number only"))
            .otherwise(pl.lit("name")))


check = check.with_columns(kind("Gene_Name").alias("paper_name_kind"))
fixable = check.filter(pl.col("paper_name_kind") != "name")
both = pl.concat([
    cands.with_columns(pl.col("gene").alias("Gene_Name"), pl.lit("missing").alias("paper_name_kind")),
    fixable.select([c for c in cands.columns if c in fixable.columns] + ["Gene_Name", "paper_name_kind"]),
], how="diagonal_relaxed")
num = ["prot_peptides_agreeing", "peptides_agreeing", "best_pct_identity", "best_evalue"]
both = both.with_columns([pl.col(c).cast(pl.Float64) for c in num])
proposed = (both.sort("prot_peptides_agreeing", descending=True, nulls_last=True)
            .group_by("gene").agg(
                pl.col("Gene_Name").first().alias("paper_name"), pl.col("paper_name_kind").first(),
                pl.col("prot_gene").drop_nulls().first().alias("proteome_name"),
                pl.col("prot_peptides_agreeing").max().alias("proteome_peptides_agreeing"),
                pl.col("sp_gene").drop_nulls().first().alias("swissprot_name"),
                pl.col("sp_protein").drop_nulls().first().alias("swissprot_protein"),
                pl.col("sp_species").drop_nulls().first().alias("swissprot_species"),
                pl.col("peptides_agreeing").max().alias("swissprot_peptides_agreeing"),
                pl.col("best_pct_identity").max().alias("swissprot_best_pct_identity"),
                pl.col("best_evalue").min().alias("swissprot_best_evalue"),
                (pl.col("contrast") + ":" + pl.col("route")).unique().sort().str.join(",").alias("reached_in"))
            .sort(["proteome_peptides_agreeing", "gene"], descending=[True, False], nulls_last=True))
on_lists = (pl.read_csv(lists_tsv, separator="\t").group_by("gene_id")
            .agg((pl.col("contrast") + ":" + pl.col("direction")).unique().sort().str.join(",").alias("on_paper_lists")))
proposed = proposed.join(on_lists, left_on="gene", right_on="gene_id", how="left")
proposed.write_csv(d / "gene_names_proposed.tsv", separator="\t")
print(proposed.group_by("paper_name_kind").agg(
    pl.len().alias("genes"), pl.col("proteome_name").is_not_null().sum().alias("proteome_name"),
    (pl.col("proteome_peptides_agreeing") >= 3).sum().alias("proteome_name_3plus_peptides"),
    pl.col("swissprot_name").is_not_null().sum().alias("swissprot_name"),
    pl.col("on_paper_lists").is_not_null().sum().alias("on_a_paper_list")))
print(proposed.filter(pl.col("on_paper_lists").is_not_null())
      .select("gene", "paper_name", "proteome_name", "proteome_peptides_agreeing", "swissprot_name", "on_paper_lists"))

real = check.filter(pl.col("paper_name_kind") == "name").unique("gene")
agree = []
for label, key, same, fam in [("new Botryllus proteome", "prot_gene", "proteome_same_name", "proteome_same_family"),
                              ("Swiss-Prot", "sp_gene", "same_name", "same_family")]:
    x = real.filter(pl.col(key).is_not_null())
    agree.append((label, x.height, (x[same] == "true").mean(), (x[fam] == "true").mean()))
    print(f"{label}: {x.height} named genes, same symbol {agree[-1][2]:.3f}, same first 3 characters {agree[-1][3]:.3f}")

CONTRASTS = ["young_2h", "young_24h", "old_24h", "young_13_49d", "young_132d", "old_41_93d"]
NAME = {"young_2h": "young 2 h", "young_24h": "young 24 h", "old_24h": "old 24 h",
        "young_13_49d": "young 13-49 d", "young_132d": "young 132 d", "old_41_93d": "old 41-93 d"}
sp = summary.filter(pl.col("route") == "splash")
pf.use_style()
fig, (a, b) = pf.figure(pf.TWO_COLUMN_MM, 55, ncols=2, gridspec_kw=dict(width_ratios=[1.3, 1]))
for y, c in enumerate(CONTRASTS):
    r = sp.filter(pl.col("contrast") == c).row(0, named=True)
    a.barh(y, r["genes_on_paper_list"], color="black", height=0.7)
    a.barh(y, r["genes_not_on_paper_list"], left=r["genes_on_paper_list"], color="#cccccc", height=0.7)
    a.text(r["genes_reached"] + 150, y, f"{r['genes_on_paper_list']} on list, "
           f"{r['genes_not_on_paper_list']:,} not", va="center")
    print(c, r["genes_on_paper_list"], r["genes_not_on_paper_list"])
a.set_xlim(0, 15000)
a.set_yticks(range(len(CONTRASTS)), [NAME[c] for c in CONTRASTS])
a.invert_yaxis()
a.set_xlabel("genes reached by changed SPLASH sequences")
a.set_title("black: on the paper's lists   light grey: not on them")
pf.panel_label(a, "a")
w = 0.38
for i, (label, n, same, fam) in enumerate(agree):
    b.bar(i - w / 2, same, width=w, color="#555555")
    b.bar(i + w / 2, fam, width=w, color="#bbbbbb")
    b.text(i - w / 2, same + 0.02, f"{same:.2f}", ha="center")
    b.text(i + w / 2, fam + 0.02, f"{fam:.2f}", ha="center")
b.set_xticks(range(len(agree)), [f"{label}\n({n:,} genes)" for label, n, _, _ in agree])
b.set_ylim(0, 1.05)
b.set_ylabel("share matching the paper's name")
b.set_title("dark grey: same symbol   light grey: same first 3 characters")
pf.panel_label(b, "b")
pf.save(fig, out_stem)
print("wrote", out_stem, "and", d / "gene_names_proposed.tsv")
