#!/usr/bin/env python3
"""What a time-split run cost so far, per process, from its Nextflow trace (notebook 280).

    trace_cost.py <trace.txt> <out.csv>

Per process and status: tasks, attempts, exits, total and longest run time, the memory
asked, and the largest peak RSS. Used at Stop A to size the seed reference run from the
Swiss-Prot pilot, the same queries and settings on a reference of nearly the same size.
"""

from __future__ import annotations

import re
import sys

import polars as pl

UNIT = {"B": 1 / 2**30, "KB": 1 / 2**20, "MB": 1 / 2**10, "GB": 1.0, "TB": 2**10}


def gb(s: str | None) -> float | None:
    m = re.match(r"^([\d.]+)\s*([KMGT]?B)$", s or "")
    return float(m[1]) * UNIT[m[2]] if m else None


def hours(s: str | None) -> float | None:
    if not s or s == "-":
        return None
    t = 0.0
    for v, u in re.findall(r"([\d.]+)(ms|d|h|m|s)", s):
        t += float(v) * {"ms": 1 / 3.6e6, "d": 24, "h": 1, "m": 1 / 60, "s": 1 / 3600}[u]
    return t


def main() -> None:
    tr = pl.read_csv(sys.argv[1], separator="\t", infer_schema_length=0)
    tr = tr.with_columns(
        process=pl.col("name").str.split(" ").list.get(0),
        ask_gb=pl.col("memory").map_elements(gb, return_dtype=pl.Float64),
        peak_gb=pl.col("peak_rss").map_elements(gb, return_dtype=pl.Float64),
        run_h=pl.col("realtime").map_elements(hours, return_dtype=pl.Float64),
    )
    out = (
        tr.group_by("process", "status")
        .agg(
            tasks=pl.len(),
            exits=pl.col("exit").unique().sort().str.join(" "),
            run_h_total=pl.col("run_h").sum().round(1),
            run_h_median=pl.col("run_h").median().round(2),
            run_h_max=pl.col("run_h").max().round(2),
            ask_gb_max=pl.col("ask_gb").max().round(0),
            peak_gb_median=pl.col("peak_gb").median().round(1),
            peak_gb_max=pl.col("peak_gb").max().round(1),
        )
        .sort("process", "status")
    )
    out.write_csv(sys.argv[2])
    with pl.Config(tbl_rows=50, tbl_width_chars=220, tbl_cols=20):
        print(out)


if __name__ == "__main__":
    main()
