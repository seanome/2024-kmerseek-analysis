"""Assign a clade to every sequence in the nb 230 Pfam seed pairs from its UniProt species
mnemonic (the _FUSHE in A0A8H5WPP7_FUSHE/132-254), via UniProt's speclist.txt and the NCBI
taxdump, and write one row per pair with both clades.

Clades: vertebrate (under Vertebrata, 7742), invertebrate (Metazoa 33208 but not Vertebrata),
fungi (4751), plant (Viridiplantae 33090), other_eukaryote, bacteria (2), archaea (2157),
virus, unknown.

Usage:
    python scripts/pfam_pair_clades.py --out /Users/olga/data/pfam/234_pfam_pair_clades.parquet
"""

from __future__ import annotations

import argparse
import re

import polars as pl

SPECLIST = "/Users/olga/data/taxonomy/speclist.txt"
NODES = "/Users/olga/data/taxonomy/nodes.dmp"
PAIRS = "/Users/olga/data/pfam/230_pfam_seed_pair_alignments.parquet"

ROOTS = [("vertebrate", 7742), ("invertebrate", 33208), ("fungi", 4751), ("plant", 33090),
         ("bacteria", 2), ("archaea", 2157), ("virus", 10239), ("eukaryote", 2759)]


def load_speclist() -> dict[str, int]:
    out = {}
    pat = re.compile(r"^([A-Z0-9]{2,5})\s+([A-Z])\s+(\d+):")
    with open(SPECLIST, errors="replace") as fh:
        for line in fh:
            m = pat.match(line)
            if m:
                out[m.group(1)] = int(m.group(3))
    return out


def load_parents() -> dict[int, int]:
    parent = {}
    with open(NODES) as fh:
        for line in fh:
            f = line.split("\t|\t")
            parent[int(f[0])] = int(f[1])
    return parent


def clade_of(taxid: int, parent: dict[int, int]) -> str:
    seen = set()
    lineage = []
    t = taxid
    while t in parent and t not in seen and t != 1:
        lineage.append(t)
        seen.add(t)
        t = parent[t]
    ls = set(lineage)
    if 7742 in ls:
        return "vertebrate"
    if 33208 in ls:
        return "invertebrate"
    for name, root in ROOTS[2:]:
        if root in ls:
            return "other_eukaryote" if name == "eukaryote" else name
    return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    spec = load_speclist()
    parent = load_parents()
    pairs = pl.read_parquet(PAIRS).select("family", "query", "target", "seqid_ali", "lali")
    mnem = lambda n: n.split("/")[0].rsplit("_", 1)[-1]
    cache: dict[str, str] = {}

    def clade(name: str) -> str:
        m = mnem(name)
        if m not in cache:
            cache[m] = clade_of(spec[m], parent) if m in spec else "unknown"
        return cache[m]

    out = pairs.with_columns(
        pl.col("query").map_elements(clade, return_dtype=pl.String).alias("query_clade"),
        pl.col("target").map_elements(clade, return_dtype=pl.String).alias("target_clade"),
    )
    out.write_parquet(args.out)
    print(out.group_by("query_clade").len().sort("len", descending=True))
    print(out.group_by("query_clade", "target_clade").len().sort("len", descending=True).head(12))


if __name__ == "__main__":
    main()
