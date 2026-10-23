"""Derive recovery requests from recorded failed jobs and per-command GNU time logs."""

import argparse
import csv
import math
import re
from pathlib import Path


def seconds(t):
    n = 0
    for v in t.split(":"):
        n = n * 60 + float(v)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", type=Path, required=True)
    ap.add_argument("--accounting", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    jobs = {}
    for line in a.accounting.read_text().splitlines():
        f = line.split("|")
        if len(f) == 6 and f[1].startswith("301-") and f[1] != "301-subset":
            jobs[f[1][4:]] = f
    rows = []
    for alphabet, job in sorted(jobs.items()):
        d = a.logs / alphabet
        if not d.is_dir():
            continue
        times = list(d.glob("*/*.time"))
        killed = [f for f in times if "signal 9" in f.read_text()]
        stage = (
            "index"
            if any(f.name == "index.time" for f in killed)
            else "search" if killed else "completed"
        )
        chunks = list(d.glob("*/search.chunk*.time"))
        completed = [f for f in chunks if "signal 9" not in f.read_text()]
        runtimes = [
            seconds(re.search(r"Elapsed.*?: (\S+)", f.read_text())[1])
            for f in completed
        ]
        peak = max(
            [
                int(m)
                for f in times
                for m in re.findall(r"Maximum resident.*?: (\d+)", f.read_text())
            ],
            default=0,
        )
        # Killed RSS is censored at 96 GiB; never treat it as the required peak.
        ask_gb = 256 if stage == "index" else 192 if stage == "search" else 0
        # Preserve 100-sequence chunk boundaries. 18 chunks; slowest successful chunk
        # times remaining chunks, x1.5 plus one hour for indexing. Floor is old 8h ask.
        predict = math.ceil(
            (max(runtimes, default=1400) * (18 - len(completed)) * 1.5 + 3600) / 3600
        )
        hours = max(8, predict) if ask_gb else 0
        rows.append(
            dict(
                alphabet=alphabet,
                previous_job_id=job[0],
                previous_state=job[2],
                previous_elapsed_s=seconds(job[3]),
                previous_memory_gib=96,
                failure_stage=stage,
                max_recorded_rss_kib=peak,
                completed_chunks=len(completed),
                requested_memory_gib=ask_gb,
                requested_hours=hours,
                request_basis=(
                    "index: >96 GiB killed; 256 GiB allows calibration growth"
                    if stage == "index"
                    else (
                        "search: >96 GiB killed; double ask, preserve chunks"
                        if stage == "search"
                        else "complete; do not resubmit"
                    )
                ),
            )
        )
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        w = csv.DictWriter(
            f, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n"
        )
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(
            r["alphabet"],
            r["failure_stage"],
            r["requested_memory_gib"],
            r["requested_hours"],
        )


if __name__ == "__main__":
    main()
