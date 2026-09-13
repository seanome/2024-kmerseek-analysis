#!/usr/bin/env python3
"""Build the local inputs the MHC-region notebooks (220-228) read.

Two products, both small enough to sit next to the notebooks:

  gencode_v50_chr6_genes.parquet  every GENCODE chr6 `gene` record with coordinates
  query_gene_map.parquet          the run's 998 human queries + query set + chromosome +
                                  chr6 coordinates + xMHC sub-region

Run after `scripts/extract_mhc.py` has been run on the cluster and its outputs pulled:

    python scripts/build_mhc_region_inputs.py \
        --gtf   gencode.v50.basic.annotation.gtf.gz \
        --midi  ~/data/qfo-pfam-region-midi-plus

or, when the chr6 GENCODE parquet already exists from an earlier build:

    python scripts/build_mhc_region_inputs.py \
        --gencode-parquet ~/data/qfo-pfam-region-midi/gencode_v50_chr6_genes.parquet \
        --midi  ~/data/qfo-pfam-region-midi-plus

The 34 queries midi-plus added off chromosome 6 (B2M, CD1A-E, MR1, KIR, LILR) get a
`chromosome` from their HGNC cytoband and no coordinates: the region notebooks only ever
place genes along chr6, and a cytoband is enough to say which chromosome a hit came from.

The GTF is GENCODE's current human release. It is a separate download rather than
`ou.HUMAN_GTF_CHR_PATCH_HAPL_SCAFF` because that file is the chr_patch_hapl_scaff build
notebook 215 needs for the MHC alt haplotypes, and this map only wants the primary
assembly -- pulling alt contigs in here would give several MHC genes two positions.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "notebooks"))
import mhc_region_utils as mu  # noqa: E402


def parse_chr6_genes(gtf: Path) -> pl.DataFrame:
    """chr6 `gene` records from a GENCODE GTF, with midpoint."""
    lf = pl.scan_csv(
        str(gtf), separator="\t", has_header=False, comment_prefix="#", quote_char=None,
        new_columns=["chrom", "source", "feature", "start", "end", "score", "strand",
                     "frame", "attr"],
    ).filter((pl.col("feature") == "gene") & (pl.col("chrom") == "chr6"))
    return (lf.select(
        gene_id=pl.col("attr").str.extract(r'gene_id "([^"]+)"', 1).str.split(".").list.get(0),
        gene_name=pl.col("attr").str.extract(r'gene_name "([^"]+)"', 1),
        gene_type=pl.col("attr").str.extract(r'gene_type "([^"]+)"', 1),
        start="start", end="end", strand="strand",
    ).with_columns(midpoint=(pl.col("start") + pl.col("end")) // 2).collect())


def pick_one_record_per_symbol(genes: pl.DataFrame) -> pl.DataFrame:
    """Collapse GENCODE symbols carrying more than one gene record down to one.

    20 chr6 symbols have two records, and 4 of them are query genes: BTN2A3P, CMAHP, HLA-H
    and LPAL2, each annotated once as a pseudogene and once as an overlapping lncRNA. Left
    alone they turn 964 queries into 968 and silently inflate every per-gene count.

    The rule is protein_coding first, then anything that is not lncRNA (for these four that
    is the pseudogene record, which is what the symbol denotes), then lowest gene_id so the
    result does not depend on file order. All four pairs overlap closely enough that the
    xMHC sub-region assignment is the same either way -- the rule decides the row count, not
    the biology.
    """
    return (genes.with_columns(
        _rank=pl.when(pl.col("gene_type") == "protein_coding").then(0)
               .when(pl.col("gene_type") != "lncRNA").then(1)
               .otherwise(2))
        .sort("_rank", "gene_id")
        .unique(subset="gene_name", keep="first", maintain_order=True)
        .drop("_rank"))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--gtf", type=Path, help="GENCODE GTF; parsed for chr6 gene records")
    p.add_argument("--gencode-parquet", type=Path,
                   help="an already-built gencode_v50_chr6_genes.parquet, instead of --gtf")
    p.add_argument("--midi", type=Path, default=mu.MIDI_DIR)
    a = p.parse_args()
    if not (a.gtf or a.gencode_parquet):
        p.error("one of --gtf or --gencode-parquet is required")

    if a.gtf:
        genes = parse_chr6_genes(a.gtf)
        genes.write_parquet(a.midi / "gencode_v50_chr6_genes.parquet")
    else:
        genes = pl.read_parquet(a.gencode_parquet)
        if a.gencode_parquet.resolve() != (a.midi / "gencode_v50_chr6_genes.parquet").resolve():
            genes.write_parquet(a.midi / "gencode_v50_chr6_genes.parquet")
    print(f"chr6 gene records: {genes.height:_} "
          f"({genes.filter(pl.col('gene_type') == 'protein_coding').height:_} protein-coding)")

    cov = pl.read_parquet(a.midi / "truth" / "human_query_covariates.parquet")
    if "query_set" not in cov.columns:  # the older midi run had chr6 only
        cov = cov.with_columns(query_set=pl.lit("chr6"))
    one = pick_one_record_per_symbol(genes)
    # Only chr6 queries may take GENCODE chr6 coordinates. A KIR symbol will never match a
    # chr6 record, but the guard makes the rule explicit rather than incidental.
    chr6 = cov.filter(pl.col("query_set") == "chr6").join(
        one.select("gene_name", "gene_id", "start", "end", "midpoint", "strand", "gene_type"),
        left_on="hgnc_symbol", right_on="gene_name", how="left")
    extras = cov.filter(pl.col("query_set") != "chr6")
    gene_map = (pl.concat([chr6, extras], how="diagonal_relaxed")
                .with_columns(mhc_subregion=mu.assign_subregion(pl.col("midpoint"))))

    # Chromosome from the HGNC cytoband, for every query. "19q13.4 alternate reference
    # locus" and plain "19q13.42" both start with the chromosome number.
    hgnc = pl.read_csv(a.midi / "hgnc_complete_set.txt", separator="\t",
                       infer_schema_length=0, quote_char=None).select(
        hgnc_symbol="symbol",
        chromosome=pl.col("location").str.extract(r"^(\d+|X|Y|MT)", 1))
    gene_map = gene_map.join(hgnc.unique(subset="hgnc_symbol"), on="hgnc_symbol", how="left")

    assert gene_map.height == cov.height, (
        f"join changed the query count: {cov.height} -> {gene_map.height}")
    assert gene_map.filter((pl.col("query_set") == "chr6") & pl.col("chromosome").is_not_null()
                           & (pl.col("chromosome") != "6")).height == 0, "chr6 query off chr6"
    gene_map.write_parquet(a.midi / "query_gene_map.parquet")

    located = gene_map.filter(pl.col("midpoint").is_not_null())
    print(f"queries: {gene_map.height:_}  located on chr6: {located.height:_}  "
          f"in xMHC: {gene_map.filter(pl.col('mhc_subregion').is_not_null()).height:_}")
    print(gene_map.group_by("query_set", "chromosome").len().sort("query_set", "chromosome"))
    print(gene_map.group_by("mhc_subregion").len().sort("len", descending=True))
    missing = gene_map.filter((pl.col("query_set") == "chr6")
                              & pl.col("midpoint").is_null())["hgnc_symbol"].to_list()
    print(f"chr6 queries with no GENCODE chr6 record under their HGNC symbol: {missing}")


if __name__ == "__main__":
    main()
