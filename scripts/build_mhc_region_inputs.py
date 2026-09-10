#!/usr/bin/env python3
"""Build the local inputs the MHC-region notebooks (220-223) read.

Two products, both small enough to sit next to the notebooks:

  gencode_v50_chr6_genes.parquet  every GENCODE chr6 `gene` record with coordinates
  chr6_query_gene_map.parquet     the midi run's 964 human queries + coordinates + sub-region

Run after `scripts/extract_mhc.py` has been run on the cluster and its outputs pulled:

    python scripts/build_mhc_region_inputs.py \
        --gtf   gencode.v50.basic.annotation.gtf.gz \
        --midi  ~/data/qfo-pfam-region-midi

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
    p.add_argument("--gtf", type=Path, required=True)
    p.add_argument("--midi", type=Path, default=mu.MIDI_DIR)
    a = p.parse_args()

    genes = parse_chr6_genes(a.gtf)
    genes.write_parquet(a.midi / "gencode_v50_chr6_genes.parquet")
    print(f"chr6 gene records: {genes.height:_} "
          f"({genes.filter(pl.col('gene_type') == 'protein_coding').height:_} protein-coding)")

    cov = pl.read_parquet(a.midi / "truth" / "human_query_covariates.parquet")
    one = pick_one_record_per_symbol(genes)
    gene_map = (cov.join(
        one.select("gene_name", "gene_id", "start", "end", "midpoint", "strand", "gene_type"),
        left_on="hgnc_symbol", right_on="gene_name", how="left")
        .with_columns(mhc_subregion=mu.assign_subregion(pl.col("midpoint"))))

    assert gene_map.height == cov.height, (
        f"join changed the query count: {cov.height} -> {gene_map.height}")
    gene_map.write_parquet(a.midi / "chr6_query_gene_map.parquet")

    located = gene_map.filter(pl.col("midpoint").is_not_null())
    print(f"queries: {gene_map.height:_}  located: {located.height:_}  "
          f"in xMHC: {gene_map.filter(pl.col('mhc_subregion').is_not_null()).height:_}")
    print(gene_map.group_by("mhc_subregion").len().sort("len", descending=True))
    missing = gene_map.filter(pl.col("midpoint").is_null())["hgnc_symbol"].to_list()
    print(f"no GENCODE chr6 record under their HGNC symbol: {missing}")


if __name__ == "__main__":
    main()
