"""Write launch resources from committed measurements; never modify tracked files."""

import csv
import math
import re
from pathlib import Path
import sys
from artifacts import sha, write_new


def measured_request(rows):
    completed = [
        r
        for r in rows
        if r["name"] == "BUILD_CORRECTED_COLLECTIONS"
        and r["status"] == "COMPLETED"
        and r["exit"] == "0"
    ]
    if not completed:
        return (
            "16 GB",
            "4h",
            "No successful grouped build in history; bounded 2000-record chunks, prior grouped audit peak 1.37 GiB, conservative four-hour model ceiling.",
        )
    memory = []
    seconds = []
    for row in completed:
        match = re.fullmatch(r"([0-9.]+) ([KMGT]?B)", row["peak_rss"])
        if not match:
            raise ValueError("Missing measured grouped-build memory")
        memory.append(
            float(match[1])
            * {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3, "TB": 1024**4}[
                match[2]
            ]
        )
        parts = re.findall(r"([0-9.]+)(ms|s|m|h|d)", row["realtime"])
        if not parts:
            raise ValueError("Missing measured grouped-build time")
        seconds.append(
            sum(
                float(n) * {"ms": 0.001, "s": 1, "m": 60, "h": 3600, "d": 86400}[u]
                for n, u in parts
            )
        )
    gb = max(4, math.ceil(2 * max(memory) / 1024**3))
    minutes = max(20, math.ceil(1.5 * max(seconds) / 300) * 5)
    return (
        f"{gb} GB",
        f"{minutes} min",
        "Same 380867-query grouped build: twice measured peak RSS with a 4-GB floor; 1.5 times measured runtime rounded up to five minutes with a 20-minute floor.",
    )


def prepare(history, output):
    output.mkdir()
    traces = sorted(history.rglob("*.trace.tsv"))
    if not traces:
        raise ValueError("No measured history")
    evidence = []
    all_rows = []
    for path in traces:
        with path.open() as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        all_rows.extend(rows)
        evidence.append(
            dict(
                file=str(path),
                sha256=sha(path),
                tasks=[
                    {
                        k: r[k]
                        for k in ("name", "status", "exit", "realtime", "peak_rss")
                    }
                    for r in rows
                ],
            )
        )
        for row in rows:
            if row["exit"] in ("137", "135", "140"):
                raise ValueError("Review resource failure before launch")
    memory, time, rationale = measured_request(all_rows)
    write_new(
        output / "resources.json",
        dict(
            evidence=evidence,
            cpus=4,
            memory=memory,
            time=time,
            rationale=rationale,
            expected_repeated_resource_failures=0,
        ),
    )
    (output / "resources.config").write_text(
        f"params {{ task_cpus = 4; task_memory = '{memory}'; task_time = '{time}'; partition = 'normal,hns' }}\nparams.code = null\n"
    )


if __name__ == "__main__":
    prepare(Path(sys.argv[1]), Path(sys.argv[2]))
