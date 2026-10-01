#!/usr/bin/env python3
"""Cut the midi-plus kmerseek region tables down to the notebook-244 cases.

For each (chosen kmerseek arm, species) among the rows of tables/244_hero_candidates.csv,
reads results/kmerseek/human_vs_<species>.<alphabet>.k<k>.lc<true|false>.regions.parquet
and keeps every row whose query is one of that arm's case proteins: all target proteins the
search hit, every region, unfiltered. Writes one parquet per table to --out-dir, and
MISSING.txt listing any table not found.

The full tables are 21 GB for the 33 arms; the cut is small enough to copy to the laptop.

    python3 scripts/extract_244_case_hits.py --results <midi-plus results/> --out-dir 244_case_hits
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
ARM_RE = re.compile(r"^kmerseek\.(?P<alpha>.+)_k(?P<k>\d+)_lc(?P<lc>True|False)$")


def table_name(arm: str, species: str) -> str:
    m = ARM_RE.match(arm)
    return (
        f"human_vs_{species}.{m['alpha']}.k{m['k']}.lc{m['lc'].lower()}.regions.parquet"
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument(
        "--candidates", type=Path, default=ROOT / "tables" / "244_hero_candidates.csv"
    )
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    c = pl.read_csv(args.candidates, infer_schema_length=None)
    want: dict[str, set[str]] = {}
    for arm, sp, q in c.select("kmerseek_chosen_arm", "species", "query").iter_rows():
        want.setdefault(table_name(arm, sp), set()).add(q)

    missing = []
    for name, queries in sorted(want.items()):
        src = args.results / "kmerseek" / name
        if not src.exists():
            missing.append(name)
            continue
        out = args.out_dir / name
        if out.exists():
            continue
        # query_name is the whole FASTA header; the accession is its second |-field.
        (
            pl.scan_parquet(src)
            .filter(
                pl.col("query_name").str.split("|").list.get(1).is_in(list(queries))
            )
            .sink_parquet(out)
        )
        n = pl.scan_parquet(out).select(pl.len()).collect().item()
        print(f"{name}: {len(queries)} queries, {n} rows")
    (args.out_dir / "MISSING.txt").write_text("".join(f"{m}\n" for m in missing))
    print(f"{len(want) - len(missing)} of {len(want)} tables cut; missing: {missing}")


if __name__ == "__main__":
    main()
