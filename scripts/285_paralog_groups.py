#!/usr/bin/env python3
"""Human paralog groups for notebook 285, from Ensembl 115 (the release GENCODE v49 matches).

Rule: two human genes are in one group when Ensembl Compara lists them as paralogs (any
homology type). A group is a connected set of such pairs. Compara builds its paralog list
from one gene tree per family, so a group is the human genes of one gene tree. Neither Ced9,
P66, nor any kmerseek result is used. A gene with no paralog is a group of one.

Writes:
  /Users/olga/data/ensembl/115/human_paralogs.tsv      (BioMart download, cached)
  notebooks/285_paralog_groups_ensembl115.csv           (one row per GENCODE v49 canonical protein)
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl
import requests

REPO = Path(__file__).resolve().parents[1]
CACHE = Path("/Users/olga/data/ensembl/115/human_paralogs.tsv")
FASTA = Path("/Users/olga/data/gencode/human/v49/gencode.v49.pc_translations.canonical.fa")
OUT = REPO / "notebooks" / "285_paralog_groups_ensembl115.csv"
# Ensembl 115 (September 2025) is the release GENCODE v49 was made with.
MART = "https://sep2025.archive.ensembl.org/biomart/martservice"
QUERY = """<?xml version="1.0" encoding="UTF-8"?><!DOCTYPE Query>
<Query virtualSchemaName="default" formatter="TSV" header="1" uniqueRows="1" datasetConfigVersion="0.6">
<Dataset name="hsapiens_gene_ensembl" interface="default">
<Filter name="biotype" value="protein_coding"/>
<Attribute name="ensembl_gene_id"/>
<Attribute name="hsapiens_paralog_ensembl_gene"/>
<Attribute name="hsapiens_paralog_orthology_type"/>
<Attribute name="hsapiens_paralog_subtype"/>
<Attribute name="hsapiens_paralog_perc_id"/>
</Dataset></Query>"""
HEADER = "Gene stable ID"


def fetch() -> None:
    """Download the paralog pairs once. BioMart serves outages as HTML with status 200."""
    if CACHE.exists():
        return
    r = requests.get(MART, params={"query": QUERY}, timeout=1800)
    r.raise_for_status()
    text = r.text
    if not text.startswith(HEADER) or "<html" in text[:2000].lower():
        sys.exit(f"BioMart did not return the TSV: {text[:300]!r}")
    if "[success]" in text[-50:]:
        text = text.rsplit("[success]", 1)[0]
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(text)


def read_proteins() -> pl.DataFrame:
    """GENCODE v49 canonical proteins: header, gene id without version, gene symbol, length."""
    rows = []
    for line in FASTA.open():
        if line.startswith(">"):
            f = line[1:].strip().split("|")
            rows.append((line[1:].strip(), f[2].split(".")[0], f[6], int(f[7])))
    return pl.DataFrame(rows, schema=["target_name", "gene_id", "gene", "length"], orient="row")


def components(nodes: list[str], edges: list[tuple[str, str]]) -> dict[str, int]:
    """Connected components by union-find; ids numbered by first appearance in `nodes`."""
    parent = {n: n for n in nodes}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    ids: dict[str, int] = {}
    return {n: ids.setdefault(find(n), len(ids)) for n in nodes}


def main() -> None:
    fetch()
    pairs = pl.read_csv(CACHE, separator="\t", infer_schema_length=0)
    pairs.columns = ["gene_id", "paralog_id", "type", "subtype", "perc_id"]
    n_genes_mart = pairs["gene_id"].n_unique()
    pairs = pairs.filter(pl.col("paralog_id").is_not_null() & (pl.col("paralog_id") != ""))
    prot = read_proteins()
    if prot["gene_id"].n_unique() != prot.height:
        sys.exit("more than one canonical protein per gene")
    nodes = prot["gene_id"].to_list()
    known = set(nodes)
    # Paralogs outside the canonical protein set cannot be scored; drop the pair, keep both
    # genes' other links.
    edges = [(a, b) for a, b in pairs.select("gene_id", "paralog_id").iter_rows()
             if a in known and b in known]
    comp = components(nodes, edges)
    prot = prot.with_columns(group_id=pl.Series([comp[g] for g in nodes], dtype=pl.Int64))
    size = prot.group_by("group_id").agg(group_size=pl.len())
    prot = prot.join(size, on="group_id").sort("group_id", "gene")
    prot.write_csv(OUT)
    print(f"BioMart rows {pairs.height:_} (genes listed {n_genes_mart:_}); "
          f"pairs inside the canonical set {len(edges):_}")
    print(f"proteins {prot.height:_}, groups {prot['group_id'].n_unique():_}")
    print(size["group_size"].value_counts().sort("group_size").head(10))
    print("largest groups:", size.sort("group_size", descending=True)["group_size"].head(5).to_list())


if __name__ == "__main__":
    main()
