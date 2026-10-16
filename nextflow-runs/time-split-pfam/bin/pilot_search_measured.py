#!/usr/bin/env python3
"""Peak memory and run time of every kmerseek search in the time-split pilot, from its
Nextflow trace, for main.nf's first asks in the full run.

    pilot_search_measured.py tables/pilot_trace/pilot_trace_launch-pilot_2026-10-15.tsv assets/pilot_search_measured.csv

The pilot searched one chunk of 440 queries (200 sample queries + 20 positive controls,
each with a decoy) against the same Swiss-Prot 2019_08 indexes the full run uses. Rows:
alphabet, k, arm, peak_rss_gb (GiB, as the trace's peak_rss), realtime_min, queries.
Only COMPLETED or CACHED searches with exit 0 are kept; a cached row carries the metrics of
the run that made it.
"""

import re
import sys

import polars as pl

PILOT_QUERIES = 440
UNIT = {"KB": 1 / 1024**2, "MB": 1 / 1024, "GB": 1.0, "TB": 1024.0}


def gib(text: str) -> float:
    num, unit = text.split()
    return float(num) * UNIT[unit]


def minutes(text: str) -> float:
    secs = {"ms": 1e-3, "s": 1, "m": 60, "h": 3600, "d": 86400}
    return (
        sum(
            float(n) * secs[u]
            for n, u in re.findall(r"(\d+(?:\.\d+)?)(ms|d|h|m|s)", text)
        )
        / 60
    )


def main() -> None:
    trace, out = sys.argv[1], sys.argv[2]
    t = pl.read_csv(trace, separator="\t", infer_schema=False)
    t = t.filter(
        pl.col("name").str.starts_with("kmerseekSearch ")
        & pl.col("status").is_in(["COMPLETED", "CACHED"])
        & (pl.col("exit") == "0")
    )
    rows = []
    for name, peak, real in t.select("name", "peak_rss", "realtime").iter_rows():
        m = re.match(r"kmerseekSearch \((\w+)\.k(\d+)\.(exact|extend)\.", name)
        rows.append(
            {
                "alphabet": m.group(1),
                "k": int(m.group(2)),
                "arm": m.group(3),
                "peak_rss_gb": round(gib(peak), 2),
                "realtime_min": round(minutes(real), 2),
                "queries": PILOT_QUERIES,
            }
        )
    df = pl.DataFrame(rows).sort("alphabet", "k", "arm")
    dup = df.filter(pl.struct("alphabet", "k", "arm").is_duplicated())
    assert dup.height == 0, dup
    df.write_csv(out)
    print(f"{df.height} searches -> {out}")


if __name__ == "__main__":
    main()
