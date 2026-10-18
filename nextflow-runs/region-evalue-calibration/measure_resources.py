#!/usr/bin/env python3
"""Size the full run's searches from what the pilot's searches used.

    measure_resources.py <pilot trace.txt> <out params.json>

Reads every COMPLETED SEARCH task in the pilot trace (one chunk of each decoy kind) and
writes the params `make run` passes with -params-file:

  search_memory_gb  1.5 x the largest peak_rss, rounded up to a whole GB
  search_time_h     2 x the longest realtime, rounded up to a whole hour (at least 1)

1.5x on memory because other chunks hold other proteins: the dark-set run's 2000-query
human chunks ranged from 0.84 to 1.10 million residues. 2x on time for the same reason
and for a busier node. Stops when a pilot chunk never completed. A chunk that failed and
then completed on a retry is used, with its failed attempts printed.
Standard library only.
"""

import csv
import json
import math
import re
import sys

UNITS = {"B": 1 / 1024**3, "KB": 1 / 1024**2, "MB": 1 / 1024, "GB": 1, "TB": 1024}


def gigabytes(field):
    m = re.fullmatch(r"([\d.]+)\s*([KMGT]?B)", field.strip())
    if field.strip() == "-":
        raise ValueError(
            "peak_rss is '-': this trace has no memory measurements (a laptop "
            "run without a container?)"
        )
    if not m:
        raise ValueError(f"cannot read a memory amount from {field!r}")
    return float(m.group(1)) * UNITS[m.group(2)]


def hours(field):
    """Nextflow's realtime: '1h 2m 3s', '45m 10s', '12.5s', '850ms'."""
    total = 0.0
    for amount, unit in re.findall(r"([\d.]+)\s*(ms|d|h|m|s)", field):
        total += (
            float(amount)
            * {"ms": 1 / 3.6e6, "s": 1 / 3600, "m": 1 / 60, "h": 1, "d": 24}[unit]
        )
    return total


def main(trace, out):
    rows = [
        r
        for r in csv.DictReader(open(trace), delimiter="\t")
        if r["process"].endswith("SEARCH")
    ]
    done = [r for r in rows if r["status"] == "COMPLETED"]
    failed = [r for r in rows if r["status"] not in ("COMPLETED", "CACHED")]
    for r in failed:
        print(
            f"  {r['tag']} attempt {r['attempt']}: {r['status']}, exit {r['exit']}, asked {r['memory']} {r['time']}"
        )
    never = {r["tag"] for r in failed} - {r["tag"] for r in done}
    if never:
        sys.exit(
            f"pilot chunks never completed: {sorted(never)}; raise search_memory_gb / "
            "search_time_h and rerun `make pilot`"
        )
    if not done:
        sys.exit(f"no completed SEARCH task in {trace}")
    peak = max(gigabytes(r["peak_rss"]) for r in done)
    longest = max(hours(r["realtime"]) for r in done)
    params = {
        "search_memory_gb": math.ceil(1.5 * peak),
        "search_time_h": max(1, math.ceil(2 * longest)),
    }
    for r in done:
        print(
            f"  {r['tag']}: peak {gigabytes(r['peak_rss']):.1f} GB, {hours(r['realtime']):.2f} h"
        )
    print(
        f"full run asks {params['search_memory_gb']} GB and {params['search_time_h']} h per search"
    )
    with open(out, "w") as fh:
        json.dump(params, fh, indent=2)
        fh.write("\n")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(*sys.argv[1:])
