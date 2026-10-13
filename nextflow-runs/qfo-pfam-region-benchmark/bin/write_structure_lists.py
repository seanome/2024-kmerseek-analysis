#!/usr/bin/env python3
"""Write the accession list the Foldseek arm needs structures for, one file per species.

Only proteins carrying a positioned Pfam domain are listed. Everything else is
undownloadable weight: a protein with no annotated domain can neither contribute a true
positive nor be transferred from, so fetching its structure buys nothing.
"""

import argparse
from pathlib import Path

import polars as pl


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--annotations", required=True, type=Path)
    p.add_argument("--outdir", required=True, type=Path)
    args = p.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    total = 0
    for path in sorted(args.annotations.glob("*_pfam_domains.parquet")):
        species = path.name.replace("_pfam_domains.parquet", "")
        acc = (
            pl.read_parquet(path)
            .filter(pl.col("has_position"))
            .select("accession")
            .unique()
            .sort("accession")["accession"]
            .to_list()
        )
        (args.outdir / f"{species}.accessions").write_text("\n".join(acc) + "\n")
        total += len(acc)
        print(f"{species:12s} {len(acc):6d}")
    print(f"{'TOTAL':12s} {total:6d}")


if __name__ == "__main__":
    main()
