#!/usr/bin/env python3
"""Size resumed nb357 tasks from retained pilot/full traces, before each launch."""

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path


def quantity(text, units):
    pairs = re.findall(r"([\d.]+)\s*([A-Za-z]+)", text)
    return sum(float(n) * units[u.lower()] for n, u in pairs) if pairs else 0


def predict(paths):
    rows = defaultdict(list)
    for path in paths:
        with open(path) as fh:
            for row in csv.DictReader(fh, delimiter="\t"):
                rows[row["process"]].append(row)
    result = {}
    for process, records in rows.items():
        successful = [r for r in records if r["status"] == "COMPLETED"]
        killed = [
            r
            for r in records
            if r["status"] == "FAILED"
            and (
                r["exit"] in ("-", "2147483647")
                or (r["exit"].isdigit() and 128 <= int(r["exit"]) <= 143)
            )
        ]
        memory = lambda r: quantity(
            r["memory"],
            {"gb": 1, "mb": 1 / 1024, "tb": 1024, "kb": 1 / 1024**2, "b": 1 / 1024**3},
        )
        seconds = lambda text: quantity(
            text, {"d": 86400, "h": 3600, "m": 60, "s": 1, "ms": 0.001}
        )
        good_mem = max((memory(r) for r in successful), default=0)
        bad_mem = max((memory(r) for r in killed), default=0)
        gb = math.ceil(max(good_mem, 2 * bad_mem if good_mem <= bad_mem else 0, 4))
        hours = math.ceil(
            max((seconds(r["time"]) for r in records), default=3600) / 3600
        )
        hours = max(
            hours,
            math.ceil(
                1.5
                * max((seconds(r["realtime"]) for r in successful), default=0)
                / 3600
            ),
        )
        result[process] = (
            gb,
            max(1, hours),
            f"{len(successful)} completed, {len(killed)} memory/time/signal failures",
        )
    # New disk-backed implementation: 2x the largest killed ask. Its runtime must
    # be measured; 24 h is an explicit conservative allowance, not a measured ETA.
    previous = result.get("alignInput", (0, 0, "no history"))
    result["alignInput"] = (
        max(96, previous[0]),
        max(24, previous[1]),
        "new disk-backed code; old implementation killed at 48 GB",
    )
    for process, (gb, hours, _) in result.items():
        if gb > 500 or hours > 168:
            raise ValueError(
                f"{process} exceeds pipeline resource ceiling: {gb} GB, {hours} h"
            )
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--history", nargs="+", required=True)
    p.add_argument("--outdir", required=True)
    a = p.parse_args()
    paths = sorted(
        {
            path
            for pattern in a.history
            for path in Path(pattern).parent.glob(Path(pattern).name)
        }
    )
    if not paths:
        p.error("no historical traces found; refusing an unmeasured resume")
    result = predict(paths)
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    config = [
        "// Generated from historical traces; no tracked file is rewritten.",
        "process {",
    ]
    report = ["process\tmemory_gb\ttime_hours\tevidence"]
    for process, (gb, hours, evidence) in sorted(result.items()):
        config.append(
            f"    withName: '{process}' {{ memory = {{ Math.min({gb} * task.attempt, 500).GB }}; time = {{ Math.min({hours} * task.attempt, 168).h }} }}"
        )
        report.append(f"{process}\t{gb}\t{hours}\t{evidence}")
    config.append("}")
    (out / "resources.config").write_text("\n".join(config) + "\n")
    (out / "resources.tsv").write_text("\n".join(report) + "\n")
    (out / "history.txt").write_text("\n".join(map(str, paths)) + "\n")
    print("\n".join(report))


if __name__ == "__main__":
    main()
