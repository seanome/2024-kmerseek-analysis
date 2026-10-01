#!/usr/bin/env python3
"""Rank each kmerseek landing call's target among everything that search hit (notebook 245).

Input: tables/245_landing_targets_to_rank.csv, one row per (region table, human query,
target protein) for every report-half feature where the setting chosen for its type lands.
For each table, reads the regions of those queries, ranks target proteins per query by the
run's rule (scripts/kmerseek_run_rank.py) and writes one row per input row with run_rank
(empty when no region of the target passes the cut) and n_ranked (proteins that pass).

    python3 scripts/rank_245_landing_targets.py --results <midi-plus results/> --out 245_landing_target_ranks.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from kmerseek_run_rank import ranked_targets  # noqa: E402


def acc(col: str) -> pl.Expr:
    return pl.col(col).str.split("|").list.get(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument(
        "--want", type=Path, default=ROOT / "tables" / "245_landing_targets_to_rank.csv"
    )
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    want = pl.read_csv(args.want)
    parts, missing = [], []
    for (table,), w in want.group_by("table", maintain_order=True):
        src = args.results / "kmerseek" / table
        if not src.exists():
            missing.append(table)
            continue
        ranks = ranked_targets(
            pl.scan_parquet(src)
            .with_columns(accession=acc("query_name"), target_acc=acc("target_name"))
            .filter(pl.col("accession").is_in(w["accession"].unique().to_list())),
            by=["accession"],
        ).collect()
        n_all = ranks.group_by("accession").agg(pl.col("n_ranked").first())
        got = (
            w.join(
                ranks.select("accession", pl.col("target_acc").alias("target"), "rank"),
                on=["accession", "target"],
                how="left",
            )
            .join(n_all, on="accession", how="left")
            .rename({"rank": "run_rank"})
        )
        parts.append(got)
        print(
            f"{table}: {w.height} calls, {got['run_rank'].null_count()} unranked",
            flush=True,
        )
    out = pl.concat(parts)
    out.write_csv(args.out)
    print(
        f"wrote {args.out}: {out.height} of {want.height} rows; missing tables: {missing}"
    )
    if missing:
        sys.exit(1)


if __name__ == "__main__":
    main()
