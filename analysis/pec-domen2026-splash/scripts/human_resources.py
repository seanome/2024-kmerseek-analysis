#!/usr/bin/env python3
"""Run time and memory of every finished task of the human pipeline (notebook 361).

Usage: human_resources.py <out.tsv> <trace.txt>...

Reads Nextflow trace files (nextflow-runs/pec-domen2026-human, launch folders run-pilot/ and
run/), keeps COMPLETED tasks, and for a task seen in several traces keeps the last. Columns:
process, tag, cpus asked, memory asked, attempt, wall time (realtime) in hours, CPU hours
(realtime x %cpu / 100), peak RSS in GB, as Nextflow measured them. Then one summary row per
process: tasks, total CPU hours, largest wall time and largest peak memory.
"""
import re
import sys

import polars as pl

out, *traces = sys.argv[1:]
UNIT = {"ms": 1e-3, "s": 1, "m": 60, "h": 3600, "d": 86400}
MEM = {"B": 1, "KB": 1e3, "MB": 1e6, "GB": 1e9, "TB": 1e12}


def secs(x):
    if x in (None, "-", ""):
        return None
    return sum(float(v) * UNIT[u] for v, u in re.findall(r"([\d.]+)(ms|s|m|h|d)", x))


def gb(x):
    if x in (None, "-", ""):
        return None
    v, u = x.split()
    return float(v) * MEM[u] / 1e9


rows = []
for t in traces:
    d = pl.read_csv(t, separator="\t", infer_schema_length=0)
    for r in d.filter(pl.col("status") == "COMPLETED").iter_rows(named=True):
        rt = secs(r["realtime"])
        cpu = float(r["%cpu"].rstrip("%")) if r["%cpu"] not in ("-", "") else None
        rows.append(dict(trace=t, complete=r["complete"], process=r["process"], tag=r["tag"],
                         cpus=int(r["cpus"]), memory_asked=r["memory"], attempt=int(r["attempt"]),
                         wall_h=round(rt / 3600, 3) if rt is not None else None,
                         cpu_h=round(rt * cpu / 100 / 3600, 3) if rt is not None and cpu is not None else None,
                         peak_rss_gb=round(gb(r["peak_rss"]), 2) if gb(r["peak_rss"]) is not None else None))
d = (pl.DataFrame(rows).sort("complete").unique(["process", "tag"], keep="last")
     .sort("process", "tag").drop("trace"))
d.write_csv(out, separator="\t")
s = (d.group_by("process").agg(pl.len().alias("tasks"), pl.col("cpu_h").sum().round(2).alias("cpu_h_total"),
                               pl.col("wall_h").max().alias("wall_h_max"), pl.col("peak_rss_gb").max().alias("peak_rss_gb_max"),
                               pl.col("attempt").max().alias("attempt_max"))
     .sort("process"))
s.write_csv(out.replace(".tsv", "_by_process.tsv"), separator="\t")
with pl.Config(tbl_rows=100, tbl_width_chars=200):
    print(s)
