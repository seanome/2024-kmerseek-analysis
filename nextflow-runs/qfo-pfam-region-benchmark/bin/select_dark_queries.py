#!/usr/bin/env python3
"""The dark stratum: human queries that hmmscan, phmmer, jackhmmer and MMseqs2 all miss.

Reads a finished run's region tables and lists every human query with no significant hit
from any of the four sequence arms against any target. That is the BHF situation at scale
(botryllus-mhc: BLAST 0, HMMer 0), and the only population on which a kmerseek region hit
cannot be a re-discovery of someone's homology inference.

Sources, in the layout the baseline processes publish under the outdir:
  regions/hmmscan/human.hmmscan.tsv.gz                     (Pfam-A hmmscan of the queries)
  regions/hmmer3_phmmer/human_vs_<species>.hmmer3_phmmer.tsv.gz
  regions/hmmer3_jackhmmer/human_vs_<species>.hmmer3_jackhmmer.tsv.gz
  regions/mmseqs2_seqseq/human_vs_<species>.mmseqs2_seqseq.tsv.gz
  regions/mmseqs2_iterative/human_vs_<species>.mmseqs2_iterative.tsv.gz
The tool is the subdirectory name. A region counts as a hit at evalue <= --max-evalue. The query universe is the run's human
FASTA, so a query with no region table entry at all is dark, not missing.

Usage:
    select_dark_queries.py --regions-dir <outdir>/regions --human-fasta <qfo>/Eukaryota/UP000005640_9606.fasta \
        --out dark_queries.txt [--max-evalue 0.01]
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import polars as pl

TOOLS = ("hmmscan", "hmmer3_phmmer", "hmmer3_jackhmmer", "mmseqs2_seqseq", "mmseqs2_iterative")


def accession_of(header: str) -> str:
    name = header[1:].split()[0]
    return name.split("|")[1] if "|" in name else name


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--regions-dir", required=True, type=Path)
    p.add_argument("--human-fasta", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--summary-out", type=Path)
    p.add_argument("--max-evalue", type=float, default=0.01)
    args = p.parse_args()

    opener = gzip.open if args.human_fasta.suffix == ".gz" else open
    with opener(args.human_fasta, "rt") as f:
        universe = {accession_of(l) for l in f if l.startswith(">")}

    lit: dict[str, set[str]] = {t: set() for t in TOOLS}
    files = sorted(args.regions_dir.glob("*/*.tsv.gz")) + sorted(args.regions_dir.glob("*/*.tsv"))
    for path in files:
        tool = path.parent.name
        if tool not in lit or path.name.endswith("_skipped.tsv"):
            continue
        lf = pl.scan_csv(path, separator="\t", infer_schema_length=10_000)
        names = lf.collect_schema().names()
        qcol = "query" if "query" in names else ("query_name" if "query_name" in names else names[0])
        ecol = "evalue" if "evalue" in names else None
        q = lf.select(pl.col(qcol).alias("q"), *( [pl.col(ecol).cast(pl.Float64).alias("e")] if ecol else []))
        if ecol:
            q = q.filter(pl.col("e") <= args.max_evalue)
        hits = q.select("q").unique().collect()["q"].to_list()
        lit[tool].update(accession_of(">" + h) for h in hits)
        print(f"{path.name}: {len(hits)} queries with a hit", flush=True)

    any_hit = set().union(*lit.values())
    dark = sorted(universe - any_hit)
    args.out.write_text("\n".join(dark) + "\n")
    summary = {"n_queries": len(universe), "n_dark": len(dark), "max_evalue": args.max_evalue,
               "lit_by_tool": {t: len(v) for t, v in lit.items()}, "tools_seen": [t for t, v in lit.items() if v]}
    if args.summary_out:
        args.summary_out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    missing = [t for t, v in lit.items() if not v]
    if missing:
        print(f"WARNING: no region table found for {missing}; the dark set is only as dark as the arms that ran")


if __name__ == "__main__":
    main()
