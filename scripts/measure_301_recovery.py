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


# hns nodes have 192000, 256000, 1024000 or 1536000 MB (sinfo, 2026-11-03). 192G is
# 196608 MB and fits no 192000 MB node; 248G (253952 MB) fits all 37 nodes of 256000 MB
# or more, where 256G (262144 MB) fits only the 15 of 1024000 MB or more.
MIN_ASK_GIB = 248


def req_mem_gib(text):
    """sacct ReqMem ("96G", "98304M", "1T") in GiB."""
    m = re.fullmatch(r"([\d.]+)([KMGT])n?c?", text)
    if m is None:
        return 0
    scale = {"K": 1 / 1024**2, "M": 1 / 1024, "G": 1, "T": 1024}[m[2]]
    return math.ceil(float(m[1]) * scale)


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
            # sacct lists jobs in id order, so the latest job of either name wins.
            jobs[f[1][4:].removeprefix("resume-")] = f
    # An alphabet with no finished chunk borrows the slowest chunk of any alphabet.
    all_runtimes = [
        seconds(re.search(r"Elapsed.*?: (\S+)", f.read_text())[1])
        for f in a.logs.glob("*/*/search.chunk*.time")
        if "signal 9" not in f.read_text()
    ]
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
        # Killed RSS is censored at the old ask; never treat it as the required peak.
        # Ask at least twice the killed ask, and at least MIN_ASK_GIB.
        prev_gib = req_mem_gib(job[4])
        ask_gb = max(MIN_ASK_GIB, 2 * prev_gib) if stage in ("index", "search") else 0
        # Preserve 100-sequence chunk boundaries. 18 chunks; slowest successful chunk
        # times remaining chunks, x1.5 plus one hour for indexing. Floor is old 8h ask.
        slowest = max(runtimes or all_runtimes, default=1400)
        predict = math.ceil((slowest * (18 - len(completed)) * 1.5 + 3600) / 3600)
        hours = max(8, predict) if ask_gb else 0
        rows.append(
            dict(
                alphabet=alphabet,
                previous_job_id=job[0],
                previous_state=job[2],
                previous_elapsed_s=seconds(job[3]),
                previous_memory_gib=prev_gib,
                failure_stage=stage,
                max_recorded_rss_kib=peak,
                completed_chunks=len(completed),
                requested_memory_gib=ask_gb,
                requested_hours=hours,
                request_basis=(
                    f"index: killed at {prev_gib} GiB; at least double, preserve chunks"
                    if stage == "index"
                    else (
                        f"search: killed at {prev_gib} GiB; at least double, preserve chunks"
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
