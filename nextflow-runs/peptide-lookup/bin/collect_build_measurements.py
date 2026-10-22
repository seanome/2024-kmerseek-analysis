"""Collect successful build and pilot timings without mixing them with predictions."""

import argparse
import csv
from pathlib import Path

from measure_indexes import read_time

p = argparse.ArgumentParser(__doc__)
p.add_argument("step4", type=Path)
p.add_argument("out", type=Path)
a = p.parse_args()
rows = []
files = [("kmerseek", "compile", a.step4 / "compile.time.txt")]
files += [
    (tool, "full_database_build", a.step4 / "comparators" / f"{tool}.time.txt")
    for tool in ("diamond", "blast")
]
files += [
    (
        "kmerseek",
        "pilot_search_memory_only" if "search_memory_only" in f.name else "pilot_index",
        f,
    )
    for f in sorted((a.step4 / "pilot").glob("*.time.txt"))
]
files += [
    ("kmerseek", "full_index_and_fit", f)
    for f in sorted((a.step4 / "full").glob("*.time.txt"))
]
for tool, operation, f in files:
    if not f.exists() or not f.stat().st_size:
        continue
    try:
        memory, seconds = read_time(f)
    except ValueError:
        continue
    rows.append(
        dict(
            tool=tool,
            operation=operation,
            record=f.name,
            elapsed_seconds=seconds,
            peak_rss_gib=memory,
        )
    )
a.out.parent.mkdir(parents=True, exist_ok=True)
with a.out.open("w") as fh:
    w = csv.DictWriter(
        fh,
        fieldnames=["tool", "operation", "record", "elapsed_seconds", "peak_rss_gib"],
        delimiter="\t",
    )
    w.writeheader()
    w.writerows(rows)
print(f"{len(rows)} successful timing records")
