"""Summarize completed pilot builds and bound full-data resource requests."""

import argparse
import csv
import re
from pathlib import Path


def read_time(path):
    text = path.read_text()
    rss = re.search(r"Maximum resident set size \(kbytes\): (\d+)", text)
    elapsed = re.search(r"Elapsed \(wall clock\) time .*?: (\S+)", text)
    status = re.search(r"Exit status: (\d+)", text)
    if not (rss and elapsed and status) or status[1] != "0":
        raise ValueError(f"incomplete or failed time record: {path}")
    seconds = 0.0
    for field in elapsed[1].split(":"):
        seconds = seconds * 60 + float(field)
    return float(rss[1]) / 1024**2, seconds


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("pilot_dir", type=Path)
    p.add_argument("out_tsv", type=Path)
    p.add_argument("--full-proteins", type=int, default=13859294)
    a = p.parse_args()
    groups = {}
    for flag in sorted(a.pilot_dir.glob("*.complete")):
        m = re.fullmatch(r"(.+)\.k(\d+)\.n(\d+)\.complete", flag.name)
        if not m:
            continue
        stem = flag.with_suffix("")
        mem, sec = read_time(Path(str(stem) + ".time.txt"))
        disk = int(Path(str(stem) + ".disk.txt").read_text().split()[0])
        groups.setdefault((m[1], int(m[2])), {})[int(m[3])] = (mem, sec, disk)
    rows = []
    for (alphabet, k), points in groups.items():
        if not {20000, 100000} <= points.keys():
            continue
        lo, hi = points[20000], points[100000]
        factor = a.full_proteins / 100000
        # Keep the fixed cost and extrapolate the measured increment; also show a
        # fully linear upper estimate. This is a prediction, never a measurement.
        inc = max(0, hi[0] - lo[0]) / 80000
        affine = hi[0] + inc * (a.full_proteins - 100000)
        upper = hi[0] * factor
        search_path = (
            a.pilot_dir / f"{alphabet}.k{k}.n100000.search_memory_only.time.txt"
        )
        search_memory, search_seconds = read_time(search_path)
        calibration_upper = max(hi[0], search_memory) * factor
        hours = hi[1] * factor * 2 / 3600
        rows.append(
            dict(
                alphabet=alphabet,
                ksize=k,
                pilot_peak_gb=hi[0],
                pilot_seconds=hi[1],
                extrapolation_factor=factor,
                full_affine_gb=affine,
                full_linear_gb=upper,
                pilot_search_peak_gb=search_memory,
                pilot_search_seconds=search_seconds,
                full_calibration_linear_gb=calibration_upper,
                full_time_limit_hours=hours,
                predicted_disk_gb=hi[2] * factor / 1e9,
                conservative_cpu_hours=hours * 8,
            )
        )
    a.out_tsv.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise SystemExit("No paired, completed pilots; no full-build requests written")
    with a.out_tsv.open("w") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(
        f"{len(rows)} paired pilots summarized; full requests require the budget check"
    )


if __name__ == "__main__":
    main()
