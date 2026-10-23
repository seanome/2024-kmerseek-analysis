#!/usr/bin/env python3
"""Size landing tasks from saved traces before each launch; keep generated asks in run/."""

import argparse
import csv
import math
import re
from pathlib import Path


def quantity(text, multipliers):
    matches = re.findall(r"([\d.]+)\s*(ms|[A-Za-z]+)", text)
    return sum(float(n) * multipliers[u] for n, u in matches if u in multipliers)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("settings", type=Path)
    a = ap.parse_args()
    history = {}
    for path in a.run_dir.glob("*trace*.txt"):
        for row in csv.DictReader(path.open(), delimiter="\t"):
            if row["process"] != "landings":
                continue
            history.setdefault(row["tag"], []).append(row)
    settings = [
        r
        for r in csv.DictReader(
            (line for line in a.settings.open() if not line.startswith("#")), delimiter="\t"
        )
    ]
    result = []
    for org in ["ciona", "zebrafish"]:
        for s in settings:
            tag = f"{org}.{s['alphabet']}.k{s['ksize']}"
            rows = history.get(tag, [])
            killed = [
                quantity(r["memory"], {"GB": 1, "MB": 1 / 1024, "TB": 1024})
                for r in rows
                if r["exit"] in ["137", "135", "2147483647", "-"]
                and r["status"] == "FAILED"
            ]
            peaks = [
                quantity(r["peak_rss"], {"GB": 1, "MB": 1 / 1024, "TB": 1024})
                for r in rows
                if r["status"] in ["CACHED", "COMPLETED"]
            ]
            times = [
                quantity(
                    r["realtime"],
                    {"h": 1, "m": 1 / 60, "s": 1 / 3600, "ms": 1 / 3600000, "d": 24},
                )
                for r in rows
            ]
            memory = max(
                16,
                math.ceil(1.2 * max(peaks, default=0)),
                math.ceil(2 * max(killed, default=0)),
            )
            if memory > 500:
                raise SystemExit(f"{tag} needs {memory} GB, above ceiling")
            result.append(
                dict(
                    tag=tag,
                    memory_gb=memory,
                    time_h=max(2, math.ceil(1.5 * max(times, default=0))),
                    basis=(
                        "twice largest killed ask"
                        if killed
                        else (
                            "1.2 x successful RSS"
                            if peaks
                            else "16 GB model, no history"
                        )
                    ),
                )
            )
    out = a.run_dir / "measured"
    out.mkdir(parents=True, exist_ok=True)
    target = out / "landings.tsv"
    with target.open("w") as fh:
        w = csv.DictWriter(fh, result[0].keys(), delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(result)
    print(target)
    for r in result:
        if r["memory_gb"] > 16:
            print(r)


if __name__ == "__main__":
    main()
